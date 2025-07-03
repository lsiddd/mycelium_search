// ==============================================
//  Importações de Módulos e Bibliotecas (Crates)
// ==============================================
// Utilitários padrão do Rust
use std::collections::{ HashMap, HashSet };
use std::fs::{ self, File };
use std::hash::BuildHasherDefault;
// FIX: Removed `SeekFrom` as it was unused.
use std::io::{ self, BufRead, BufReader, BufWriter, Seek, Write };
use std::path::{ Path, PathBuf };
use std::sync::atomic::{ AtomicUsize, Ordering };
use std::sync::Mutex;
use std::time::Instant;
use std::cell::RefCell;

// Bibliotecas de terceiros (externas)
use clap::Parser;
use indicatif::{ ProgressBar, ProgressStyle };
use itertools::Itertools;
// FIX: Removed `debug` as it was unused.
use log::{ info, warn };
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use regex::Regex;
use rust_stemmers::{ Algorithm, Stemmer };
use serde::{ Deserialize, Serialize };
use zstd::stream::{ Decoder, Encoder };

// Esta é a tecnologia central que nos permite acessar os dados do índice no disco
// sem carregá-los inteiramente na memória RAM.
use memmap2::Mmap;

// ==============================
//  Otimizações e Apelidos de Tipos
// ==============================
type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<ahash::AHasher>>;

// ==============================
//  Constantes e Variáveis Estáticas
// ==============================
const K1: f64 = 1.5;
const B: f64 = 0.75;
const CHUNK_SIZE: usize = 20_000;

thread_local! {
    static STEMMER: RefCell<Stemmer> = RefCell::new(Stemmer::create(Algorithm::English));
}

// ==============================
//  Estruturas de Dados do Índice (em Memória e em Disco)
// ==============================
type TermId = u32;

// REESTRUTURADO: Esta estrutura permanece a mesma, mas agora deve ser `Copy` para
// podermos fazer "casts" de slices de bytes para slices dela de forma segura.
// A anotação `#[repr(C)]` garante que o layout da memória seja previsível.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[repr(C)]
struct TermPosting {
    doc_id: usize,
    // REESTRUTURADO: Para simplificar o formato em disco, o `TermPosting` não irá mais
    // armazenar as posições diretamente. As posições serão armazenadas em um arquivo separado.
    // Esta é uma simplificação para o exemplo, mas em um sistema real, você poderia ter um
    // arquivo positions.dat e adicionar um `positions_offset` aqui.
    // Por enquanto, vamos focar em resolver o problema da memória dos postings.
    // A frequência do termo é o que o BM25 precisa, então vamos armazená-la.
    term_frequency: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[repr(C)]
struct DocumentMetadata {
    file_id: usize,
    line_number: usize,
    doc_len: u32,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct TermDictionary {
    map: AHashMap<String, TermId>,
    rev_map: Vec<String>,
}

impl TermDictionary {
    fn get_or_insert(&mut self, term: &str) -> TermId {
        if let Some(id) = self.map.get(term) {
            return *id;
        }
        let new_id = self.rev_map.len() as TermId;
        self.map.insert(term.to_string(), new_id);
        self.rev_map.push(term.to_string());
        new_id
    }
}

// REESTRUTURADO: Esta é a estrutura que contém os metadados do índice.
// Ela é PEQUENA e será carregada completamente na memória.
#[derive(Debug, Serialize, Deserialize)]
struct IndexMetadata {
    file_table: Vec<PathBuf>,
    stop_words: HashSet<String>,
    total_docs: usize,
    avg_doc_len: f64,
    term_dictionary: TermDictionary,
    doc_frequencies: AHashMap<TermId, usize>,
}

// REESTRUTURADO: O `ShardedIndex` antigo foi renomeado para `IndexBuilder`.
// Seu único propósito agora é coletar todos os dados em memória durante a fase de indexação
// antes de serem escritos no formato de disco pelo `IndexWriter`.
#[derive(Debug)]
struct IndexBuilder {
    // Note: The types for these HashMaps will be inferred from their usage later
    shards_postings: Vec<AHashMap<TermId, Vec<(usize, u32)>>>, // (doc_id, term_frequency)
    shards_docs: Vec<AHashMap<usize, DocumentMetadata>>,
    total_docs: AtomicUsize,
    term_dictionary: Mutex<TermDictionary>,
    file_table: Vec<PathBuf>,
    stop_words: HashSet<String>,
    num_shards: usize,
}

// REESTRUTURADO: Introdução do `IndexReader`.
// Esta estrutura representa um índice aberto para busca. Note como ela NÃO contém
// as listas de postings. Em vez disso, ela tem mapeamentos de memória (`Mmap`)
// para os arquivos no disco. Esta é a chave para o baixo uso de RAM.
#[derive(Debug)]
struct IndexReader {
    metadata: IndexMetadata,
    docs: Mmap,
    postings: Mmap,
    postings_offsets: Vec<u64>,
}

// REESTRUTURADO: Introdução do `IndexWriter`.
// Responsável por pegar os dados coletados pelo `IndexBuilder` e escrevê-los
// no formato de diretório de índice otimizado para o `IndexReader`.
struct IndexWriter;

// ==========================================================
//  Implementação do `IndexBuilder` (Coleta de Dados)
// ==========================================================
impl IndexBuilder {
    pub fn new(num_shards: usize) -> Self {
        IndexBuilder {
            shards_postings: (0..num_shards).map(|_| AHashMap::default()).collect(),
            shards_docs: (0..num_shards).map(|_| AHashMap::default()).collect(),
            total_docs: AtomicUsize::new(0),
            term_dictionary: Mutex::new(TermDictionary::default()),
            file_table: Vec::new(),
            stop_words: Self::get_stop_words(),
            num_shards,
        }
    }

    pub fn index_directory(&mut self, dir_path: &Path) -> io::Result<()> {
        info!("Buscando arquivos .txt no diretório: {:?}", dir_path);
        let files: Vec<_> = fs
            ::read_dir(dir_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(
                |path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt")
            )
            .collect();

        self.file_table = files;
        info!("Encontrados {} arquivos para indexar.", self.file_table.len());

        let pb = ProgressBar::new(self.file_table.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
                )
                .unwrap()
                .progress_chars("#>-")
        );

        type PostingsMap = AHashMap<TermId, Vec<(usize, u32)>>;
        type DocsMap = AHashMap<usize, DocumentMetadata>;

        let temp_shard_data: Vec<_> = (0..self.num_shards)
            // Use `::default()` which provides the compiler with the full type information.
            .map(|_| Mutex::new((PostingsMap::default(), DocsMap::default()))) // (postings, docs)
            .collect();

        for (file_id, path) in self.file_table.iter().enumerate() {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                for (chunk_idx, chunk) in reader
                    .lines()
                    .chunks(CHUNK_SIZE)
                    .into_iter()
                    .enumerate() {
                    let lines_chunk: Vec<String> = chunk.filter_map(Result::ok).collect();
                    let base_line_num = chunk_idx * CHUNK_SIZE;

                    lines_chunk
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(i, line)| {
                            let doc_id = self.total_docs.fetch_add(1, Ordering::Relaxed);
                            let shard_index = doc_id % self.num_shards;
                            let line_number = base_line_num + i + 1;

                            STEMMER.with(|stemmer_cell| {
                                let stemmer = stemmer_cell.borrow();
                                let tokens = self.tokenize_and_stem(&line, &stemmer);
                                let doc_len = tokens.len() as u32;

                                let mut term_counts: AHashMap<TermId, u32> = AHashMap::default();
                                for token in tokens {
                                    let term_id = {
                                        let mut dict = self.term_dictionary.lock().unwrap();
                                        dict.get_or_insert(&token)
                                    };
                                    *term_counts.entry(term_id).or_insert(0) += 1;
                                }

                                let mut shard_lock = temp_shard_data[shard_index].lock().unwrap();
                                let (ref mut postings, ref mut docs) = *shard_lock;

                                docs.insert(doc_id, DocumentMetadata {
                                    file_id,
                                    line_number,
                                    doc_len,
                                });
                                for (term_id, count) in term_counts {
                                    postings.entry(term_id).or_default().push((doc_id, count));
                                }
                            });
                        });
                }
            } else {
                warn!("Não foi possível abrir o arquivo: {:?}", path);
            }
            pb.inc(1);
        }
        pb.finish_with_message("Coleta de dados em memória concluída.");

        // Mesclar dados temporários
        for (i, mutex) in temp_shard_data.into_iter().enumerate() {
            let (postings, docs) = mutex.into_inner().unwrap();
            self.shards_postings[i] = postings;
            self.shards_docs[i] = docs;
        }

        Ok(())
    }

    // Métodos `tokenize_and_stem` e `get_stop_words` (com correção no stemmer)
    fn tokenize_and_stem(&self, text: &str, stemmer: &Stemmer) -> Vec<String> {
        static TOKEN_FINDER_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\w.-]+").unwrap());
        TOKEN_FINDER_REGEX.find_iter(text)
            .map(|mat| mat.as_str().to_lowercase())
            .filter(|s| !s.is_empty() && !self.stop_words.contains(s))
            // FIX (E0308): Pass a reference `&s` instead of an owned `String` `s`.
            .map(|s| stemmer.stem(&s).into_owned())
            .collect()
    }

    fn get_stop_words() -> HashSet<String> {
        [
            "a",
            "an",
            "the",
            "in",
            "on",
            "of",
            "for",
            "to",
            "with",
            "is",
            "are",
            "was",
            "were",
            "at",
            "by",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "am",
        ]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

// ==========================================================
//  Implementação do `IndexWriter` (Escrita para o Disco)
// ==========================================================

impl IndexWriter {
    pub fn write(builder: IndexBuilder, index_path: &Path) -> io::Result<()> {
        if index_path.exists() {
            fs::remove_dir_all(index_path)?;
        }
        fs::create_dir_all(index_path)?;
        info!("Iniciando a escrita do índice no diretório: {:?}", index_path);

        let total_docs = builder.total_docs.load(Ordering::Relaxed);
        let term_dictionary = builder.term_dictionary.into_inner().unwrap();
        let num_terms = term_dictionary.rev_map.len();

        // 1. Mesclar documentos e calcular o comprimento médio e as frequências de documentos
        info!("Mesclando metadados dos documentos...");
        let mut all_docs =
            vec![DocumentMetadata { file_id: 0, line_number: 0, doc_len: 0 }; total_docs];
        let mut total_doc_len_sum: u64 = 0;
        for shard_docs in builder.shards_docs {
            for (doc_id, metadata) in shard_docs {
                all_docs[doc_id] = metadata;
                total_doc_len_sum += metadata.doc_len as u64;
            }
        }
        let avg_doc_len = if total_docs > 0 {
            (total_doc_len_sum as f64) / (total_docs as f64)
        } else {
            0.0
        };

        // Escrever o arquivo de metadados dos documentos (`docs.dat`)
        let docs_path = index_path.join("docs.dat");
        let mut docs_file = BufWriter::new(File::create(&docs_path)?);
        // REESTRUTURADO: Usamos `as_bytes` para uma escrita de baixo nível, muito mais rápida
        // do que a serialização individual. Isso é seguro por causa do `#[repr(C)]`.
        let docs_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                all_docs.as_ptr() as *const u8,
                all_docs.len() * std::mem::size_of::<DocumentMetadata>()
            )
        };
        docs_file.write_all(docs_bytes)?;
        info!("Arquivo de metadados de documentos salvo em {:?}", docs_path);

        // 2. Mesclar postings, ordenar por doc_id e calcular frequências
        info!("Mesclando e ordenando listas de postings...");
        let mut all_postings: Vec<Vec<TermPosting>> = vec![Vec::new(); num_terms];
        let mut doc_frequencies = AHashMap::default();
        for shard_postings in builder.shards_postings {
            for (term_id, postings) in shard_postings {
                let term_postings = all_postings.get_mut(term_id as usize).unwrap();
                for (doc_id, tf) in postings {
                    term_postings.push(TermPosting { doc_id, term_frequency: tf });
                }
            }
        }

        // Ordenar postings em paralelo e calcular frequências
        all_postings
            .par_iter_mut()
            .enumerate()
            .for_each(|(_term_id, postings)| {
                postings.sort_unstable_by_key(|p| p.doc_id);
                // Esta parte da frequência precisa ser feita em um segundo passo single-threaded.
            });
        for (term_id, postings) in all_postings.iter().enumerate() {
            if !postings.is_empty() {
                doc_frequencies.insert(term_id as TermId, postings.len());
            }
        }

        // 3. Escrever os arquivos de postings (`postings.dat`) e offsets (`postings.off`)
        let postings_path = index_path.join("postings.dat");
        let offsets_path = index_path.join("postings.off");
        let mut postings_file = BufWriter::new(File::create(&postings_path)?);
        let mut offsets_file = BufWriter::new(File::create(&offsets_path)?);
        let mut postings_offsets = vec![0u64; num_terms];

        info!("Escrevendo arquivos de postings e offsets...");
        for (term_id, postings) in all_postings.iter().enumerate() {
            if postings.is_empty() {
                continue;
            }
            let offset = postings_file.stream_position()?;
            postings_offsets[term_id] = offset;

            // Escreve os postings como um bloco de bytes
            let postings_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    postings.as_ptr() as *const u8,
                    postings.len() * std::mem::size_of::<TermPosting>()
                )
            };
            postings_file.write_all(postings_bytes)?;
        }

        // Escrever os offsets
        let offsets_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                postings_offsets.as_ptr() as *const u8,
                postings_offsets.len() * std::mem::size_of::<u64>()
            )
        };
        offsets_file.write_all(offsets_bytes)?;
        info!("Arquivos de postings e offsets salvos em {:?} e {:?}", postings_path, offsets_path);

        // 4. Escrever o arquivo principal de metadados (`meta.bin`)
        let metadata = IndexMetadata {
            file_table: builder.file_table,
            stop_words: builder.stop_words,
            total_docs,
            avg_doc_len,
            term_dictionary,
            doc_frequencies,
        };

        let meta_path = index_path.join("meta.bin");
        let meta_file = File::create(&meta_path)?;
        let buffered_writer = BufWriter::new(meta_file);
        let mut encoder = Encoder::new(buffered_writer, 3)?;
        // FIX (E0277): Manually map the bincode error to an io::Error.
        bincode
            ::serialize_into(&mut encoder, &metadata)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        encoder.finish()?;
        info!("Arquivo de metadados principal salvo em {:?}", meta_path);

        Ok(())
    }
}

// ==========================================================
//  Implementação do `IndexReader` (Busca com mmap)
// ==========================================================
impl IndexReader {
    // REESTRUTURADO: A nova função `load` agora é `open`, para refletir que não estamos
    // carregando tudo, mas sim abrindo os arquivos para acesso.
    pub fn open(index_path: &Path) -> io::Result<Self> {
        info!("Abrindo o índice de {:?}. Apenas metadados serão carregados na RAM.", index_path);

        // 1. Carregar o arquivo de metadados (pequeno)
        let meta_path = index_path.join("meta.bin");
        let meta_file = File::open(&meta_path)?;
        let buffered_reader = BufReader::new(meta_file);
        let decoder = Decoder::new(buffered_reader)?;
        let metadata: IndexMetadata = bincode
            ::deserialize_from(decoder)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // 2. Carregar o arquivo de offsets dos postings (relativamente pequeno)
        let offsets_path = index_path.join("postings.off");
        let offsets_data = fs::read(offsets_path)?;
        let postings_offsets: Vec<u64> = unsafe {
            let mut vec = Vec::with_capacity(offsets_data.len() / std::mem::size_of::<u64>());
            let ptr = offsets_data.as_ptr() as *const u64;
            vec.set_len(offsets_data.len() / std::mem::size_of::<u64>());
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), vec.len());
            vec
        };

        // 3. Mapear em memória os arquivos grandes (docs e postings)
        let docs_path = index_path.join("docs.dat");
        let docs_file = File::open(&docs_path)?;
        let docs = unsafe { Mmap::map(&docs_file)? };

        let postings_path = index_path.join("postings.dat");
        let postings_file = File::open(&postings_path)?;
        let postings = unsafe { Mmap::map(&postings_file)? };

        info!("Índice aberto com sucesso. Prótons para busca.");

        Ok(IndexReader {
            metadata,
            docs,
            postings,
            postings_offsets,
        })
    }

    // REESTRUTURADO: Obtém a lista de postings de um termo. Em vez de um lookup em HashMap,
    // ele encontra o slice de bytes relevante no arquivo mapeado em memória e o
    // converte para um slice de `&[TermPosting]`. Isso é extremamente rápido e não aloca memória.
    fn get_postings_for_term(&self, term_id: TermId) -> Option<&[TermPosting]> {
        let term_idx = term_id as usize;
        if term_idx >= self.postings_offsets.len() {
            return None;
        }

        let start_offset = self.postings_offsets[term_idx] as usize;

        // FIX: Removed the unused `end_offset` variable and its logic block.
        // The second `end_offset` declaration correctly handles finding the next valid offset.
        // Encontra o primeiro offset não-zero a partir do proximo
        let end_offset = self.postings_offsets
            .get(term_idx + 1..)?
            .iter()
            .find(|&&offset| offset > 0)
            .map(|&offset| offset as usize)
            .unwrap_or(self.postings.len());

        if start_offset >= end_offset {
            return None;
        }

        let postings_bytes = &self.postings[start_offset..end_offset];
        let postings: &[TermPosting] = unsafe {
            std::slice::from_raw_parts(
                postings_bytes.as_ptr() as *const TermPosting,
                postings_bytes.len() / std::mem::size_of::<TermPosting>()
            )
        };
        Some(postings)
    }

    // REESTRUTURADO: Obtém os metadados de um documento específico.
    fn get_doc_metadata(&self, doc_id: usize) -> Option<&DocumentMetadata> {
        let size = std::mem::size_of::<DocumentMetadata>();
        let start = doc_id * size;
        let end = start + size;
        if end > self.docs.len() {
            return None;
        }
        let doc_bytes = &self.docs[start..end];
        let doc: &DocumentMetadata = unsafe { &*(doc_bytes.as_ptr() as *const DocumentMetadata) };
        Some(doc)
    }

    pub fn search(&self, query: &str) -> Vec<(PathBuf, usize, f64)> {
        let stemmer = Stemmer::create(Algorithm::English);
        let query_token_strings = self.tokenize_and_stem(query, &stemmer);

        let query_token_ids: Vec<TermId> = query_token_strings
            .iter()
            .filter_map(|token| self.metadata.term_dictionary.map.get(token).copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if query_token_ids.is_empty() {
            info!("Nenhum termo da consulta foi encontrado no dicionário.");
            return Vec::new();
        }

        let idf_map: AHashMap<TermId, f64> = query_token_ids
            .par_iter()
            .map(|&term_id| {
                let doc_freq = self.metadata.doc_frequencies.get(&term_id).copied().unwrap_or(0);
                (term_id, self.calculate_idf(doc_freq))
            })
            .collect();

        // A busca agora itera sobre os termos e acessa os postings via mmap.
        let mut doc_scores: AHashMap<usize, f64> = AHashMap::default();
        for &token_id in &query_token_ids {
            if
                let (Some(postings), Some(idf)) = (
                    self.get_postings_for_term(token_id),
                    idf_map.get(&token_id),
                )
            {
                for posting in postings {
                    let score = self.calculate_bm25(posting, *idf);
                    *doc_scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }

        // REESTRUTURADO: A lógica de ranking de proximidade precisaria ser readaptada
        // para um novo formato de arquivo de posições. Para manter este exemplo focado
        // no problema de memória, a busca por proximidade foi removida temporariamente.
        // Adicioná-la de volta envolveria criar um `positions.dat` e `positions.off`.

        let mut all_results: Vec<_> = doc_scores
            .into_par_iter()
            .filter_map(|(doc_id, score)| {
                self.get_doc_metadata(doc_id).map(|metadata| {
                    let file_path = self.metadata.file_table[metadata.file_id].clone();
                    (file_path, metadata.line_number, score)
                })
            })
            .collect();

        all_results.par_sort_unstable_by(|a, b|
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        );
        all_results
    }

    fn calculate_bm25(&self, posting: &TermPosting, idf: f64) -> f64 {
        let doc_metadata = self.get_doc_metadata(posting.doc_id).unwrap();
        let doc_len = doc_metadata.doc_len as f64;
        let tf = posting.term_frequency as f64;
        let numerator = tf * (K1 + 1.0);
        let denominator = tf + K1 * (1.0 - B + B * (doc_len / self.metadata.avg_doc_len));
        idf * (numerator / denominator)
    }

    fn calculate_idf(&self, doc_freq: usize) -> f64 {
        let total_docs = self.metadata.total_docs as f64;
        ((total_docs - (doc_freq as f64) + 0.5) / ((doc_freq as f64) + 0.5) + 1.0).ln()
    }

    // Métodos `tokenize_and_stem` e `get_stop_words` (com correção no stemmer)
    fn tokenize_and_stem(&self, text: &str, stemmer: &Stemmer) -> Vec<String> {
        static TOKEN_FINDER_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\w.-]+").unwrap());
        TOKEN_FINDER_REGEX.find_iter(text)
            .map(|mat| mat.as_str().to_lowercase())
            .filter(|s| !s.is_empty() && !self.metadata.stop_words.contains(s))
            // FIX (E0308): Pass a reference `&s` instead of an owned `String` `s`.
            .map(|s| stemmer.stem(&s).into_owned())
            .collect()
    }
}

fn get_line_from_file(file_path: &Path, line_number: usize) -> io::Result<Option<String>> {
    if line_number == 0 {
        return Ok(None);
    }
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    Ok(
        reader
            .lines()
            .nth(line_number - 1)
            .transpose()?
    )
}

// ==============================
//  Lógica Principal da Aplicação (CLI)
// ==============================

// Definições de `Cli` e `Commands` permanecem as mesmas.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = None,
    help_template = "\
{name} {version}
{author-with-newline}
{about-with-newline}
{usage-heading} {usage}

{all-args}
"
)]
struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    Index {
        #[arg(value_name = "DIRETÓRIO")]
        dir_path: PathBuf,
        // REESTRUTURADO: O output agora é um diretório, não um arquivo.
        #[arg(short, long, value_name = "DIRETÓRIO_DE_ÍNDICE", default_value = "index.myc")]
        output: PathBuf,
    },
    Search {
        #[arg(value_name = "CONSULTA")]
        query: String,
        #[arg(short, long, value_name = "DIRETÓRIO_DE_ÍNDICE", default_value = "index.myc")]
        index_path: PathBuf,
        #[arg(short, long, value_name = "ARQUIVO_DE_SAÍDA")]
        output: Option<PathBuf>,
    },
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    // Configuração do logger (sem alterações)
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        _ => log::LevelFilter::Debug,
    };
    env_logger::Builder::new().filter_level(log_level).format_timestamp_secs().init();

    match cli.command {
        Commands::Index { dir_path, output } => {
            let num_shards = num_cpus::get();
            info!("Iniciando indexação com {} shards (núcleos de CPU)...", num_shards);
            let start = Instant::now();
            let mut builder = IndexBuilder::new(num_shards);
            builder.index_directory(&dir_path)?;
            info!("Coleta de dados em memória concluída em {:?}.", start.elapsed());

            let write_start = Instant::now();
            IndexWriter::write(builder, &output)?;
            // REESTRUTURADO: Calcular o tamanho total do diretório do índice.
            let total_size: u64 = fs
                ::read_dir(&output)?
                .filter_map(Result::ok)
                .filter_map(|entry| entry.metadata().ok())
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len())
                .sum();

            println!(
                "Índice salvo em {:?} (Tamanho total: {:.2} MB) em {:?}.",
                output,
                (total_size as f64) / 1_048_576.0,
                write_start.elapsed()
            );
        }
        Commands::Search { query, index_path, output } => {
            // REESTRUTURADO: Verificar se o diretório do índice (ou o meta.bin) existe.
            if !index_path.join("meta.bin").exists() {
                eprintln!(
                    "Erro: Diretório de índice inválido ou não encontrado em {:?}. Por favor, execute o comando 'index' primeiro.",
                    index_path
                );
                return Ok(());
            }

            let load_start = Instant::now();
            // REESTRUTURADO: Usa o IndexReader para abrir o índice.
            let index_reader = IndexReader::open(&index_path)?;
            info!(
                "Índice aberto em {:?} (operação de baixo custo de memória).",
                load_start.elapsed()
            );

            info!("Buscando por: '{}'", query);
            let search_start = Instant::now();
            let results = index_reader.search(&query);
            let search_duration = search_start.elapsed();

            // Lógica de exibição/salvamento de resultados (sem grandes alterações)
            // ... (o código restante da função main é praticamente idêntico)
            if let Some(out_path) = output {
                println!(
                    "Encontrados {} resultados em {:?}. Salvando em {}...",
                    results.len(),
                    search_duration,
                    out_path.display()
                );
                let file = File::create(&out_path)?;
                let mut writer = BufWriter::new(file);
                writeln!(writer, "Resultados da busca para a consulta: '{}'", query)?;
                writeln!(writer, "Encontrados {} resultados.\n", results.len())?;
                for (file_path, line_number, score) in results.iter() {
                    let line_content = get_line_from_file(file_path, *line_number)?.unwrap_or_else(
                        || "[Não foi possível ler a linha]".to_string()
                    );
                    writeln!(
                        writer,
                        "Pontuação: {:.4}, Arquivo: \"{}\", Linha: {}\n > {}\n",
                        score,
                        file_path.display(),
                        line_number,
                        line_content.trim()
                    )?;
                }
                println!("Resultados salvos com sucesso em {}.", out_path.display());
            } else {
                if !results.is_empty() {
                    println!(
                        "\nEncontrados {} resultados em {:?}:",
                        results.len(),
                        search_duration
                    );
                    for (file_path, line_number, score) in results.iter().take(10) {
                        let line_content = get_line_from_file(
                            file_path,
                            *line_number
                        )?.unwrap_or_else(|| "[Não foi possível ler a linha]".to_string());
                        println!(
                            "  - Pontuação: {:.4}, Arquivo: \"{}\", Linha: {}",
                            score,
                            file_path.display(),
                            line_number
                        );
                        println!("    > {}", line_content.trim());
                    }
                } else {
                    println!("\nNenhum resultado encontrado em {:?}.", search_duration);
                }
            }
        }
    }

    Ok(())
}
