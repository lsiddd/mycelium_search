// ==============================================
//  Importações de Módulos e Bibliotecas (Crates)
// ==============================================
// Utilitários padrão do Rust
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::hash::BuildHasherDefault;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use std::cell::RefCell;

// Bibliotecas de terceiros (externas)
use clap::Parser; // MELHORIA: Usado para uma análise de argumentos CLI robusta.
use indicatif::{ProgressBar, ProgressStyle}; // Para barras de progresso visuais.
use itertools::Itertools; // Traz métodos extras para iteradores.
use log::{info, warn, debug}; // Para registrar mensagens de log (info, warn, debug).
use once_cell::sync::Lazy; // Para inicialização preguiçosa de estáticos.
use rayon::prelude::*; // Traits para paralelismo com Rayon.
use rayon::slice::ParallelSliceMut; // OTIMIZAÇÃO: Para ordenação paralela de slices.
use regex::Regex; // Para compilação e uso de expressões regulares.
use rust_stemmers::{Algorithm, Stemmer}; // Para o processo de stemming de palavras.
use serde::{Deserialize, Serialize}; // Traits para serialização e desserialização.
use zstd::stream::{Decoder, Encoder}; // Para compressão/descompressão com Zstandard.

// ==============================
//  Otimizações e Apelidos de Tipos (Type Aliases)
// ==============================

// OTIMIZAÇÃO: Usa o AHash, um algoritmo de hashing muito mais rápido, para todos os HashMaps.
// Isso acelera significativamente as operações de inserção e busca em mapas de hash,
// que são centrais para o desempenho do índice invertido.
type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<ahash::AHasher>>;

// ==============================
//  Constantes e Variáveis Estáticas
// ==============================
// Constantes para o algoritmo de pontuação BM25.
const K1: f64 = 1.5;  // Controla a saturação da frequência do termo (TF).
const B: f64 = 0.75; // Controla o impacto do comprimento do documento na pontuação.

// Define o tamanho do bloco (chunk) de linhas a serem processadas em paralelo.
// Um bom tamanho de chunk balanceia a sobrecarga de agendamento de threads com a localidade de dados.
const CHUNK_SIZE: usize = 20_000;

// Compila a expressão regular para tokenização apenas uma vez usando `Lazy`.
// `Lazy` garante que a compilação cara do Regex aconteça apenas na primeira vez que for acessado.
// O Regex divide o texto por espaços em branco ou pelos caracteres '|', ':', '/'.
static TOKENIZER_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s:|/]+").unwrap());

// `thread_local!` cria uma instância do `Stemmer` para cada thread.
// Isso evita a necessidade de usar Mutexes para compartilhar o stemmer,
// já que a estrutura `Stemmer` não é `Sync` (segura para compartilhamento entre threads).
// `RefCell` permite mutabilidade interna de forma segura dentro de uma única thread.
thread_local! {
    static STEMMER: RefCell<Stemmer> = RefCell::new(Stemmer::create(Algorithm::English));
}

// ==============================
//  Estruturas de Dados do Índice
// ==============================

/// Define um identificador único para cada termo (palavra) no dicionário.
/// Usar um `u32` é mais eficiente em termos de memória do que armazenar strings repetidamente.
type TermId = u32;

/// Representa a ocorrência de um termo em um documento específico.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct TermPosting {
    /// O ID do documento onde o termo aparece.
    doc_id: usize,
    /// Uma lista de posições (índices de token) onde o termo ocorre dentro do documento.
    positions: Vec<u32>,
}

/// Armazena metadados sobre cada documento (neste caso, cada linha de um arquivo).
#[derive(Debug, Serialize, Deserialize, Clone)]
struct DocumentMetadata {
    /// O ID do arquivo ao qual este documento pertence (um índice na `file_table`).
    file_id: usize,
    /// O número da linha dentro do arquivo original.
    line_number: usize,
    /// O comprimento total do documento em número de tokens.
    doc_len: u32,
}

/// O dicionário de termos, que mapeia strings de termos para `TermId`s e vice-versa.
#[derive(Debug, Serialize, Deserialize, Default)]
struct TermDictionary {
    /// Mapeia uma string de termo (ex: "gato") para seu `TermId` (ex: 42).
    map: AHashMap<String, TermId>, // OTIMIZAÇÃO: Usa AHashMap.
    /// Mapeamento reverso: um vetor onde o índice é o `TermId` e o valor é a string do termo.
    rev_map: Vec<String>,
}

impl TermDictionary {
    /// Obtém o `TermId` de um termo. Se o termo não existir, ele é inserido
    /// no dicionário e um novo `TermId` é retornado.
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

/// Uma "shard" (partição) do índice. O índice principal é dividido em várias shards
/// para permitir a indexação e a busca em paralelo.
#[derive(Debug, Serialize, Deserialize)]
struct IndexShard {
    /// O índice invertido para esta shard: mapeia `TermId` para uma lista de `TermPosting`s.
    postings: AHashMap<TermId, Vec<TermPosting>>, // OTIMIZAÇÃO: Usa AHashMap.
    /// Metadados para os documentos contidos nesta shard.
    docs: AHashMap<usize, DocumentMetadata>, // OTIMIZAÇÃO: Usa AHashMap.
    /// A soma dos comprimentos de todos os documentos nesta shard.
    total_doc_len: u64,
    /// O comprimento médio dos documentos nesta shard, usado no cálculo do BM25.
    avg_doc_len: f64,
}

/// A estrutura principal do índice, que contém todas as shards e dados globais.
#[derive(Debug, Serialize, Deserialize)]
struct ShardedIndex {
    /// O conjunto de todas as partições (shards) do índice.
    shards: Vec<IndexShard>,
    /// Uma tabela que mapeia um `file_id` para o caminho do arquivo (`PathBuf`).
    file_table: Vec<PathBuf>,
    /// Um conjunto de "stop words" (palavras comuns como "o", "a", "de") a serem ignoradas.
    stop_words: HashSet<String>,
    /// O número total de documentos em todo o índice.
    total_docs: usize,
    /// O dicionário de termos global para todo o índice.
    term_dictionary: TermDictionary,
    // OTIMIZAÇÃO: Armazena a frequência de documentos global para cada termo.
    // Isso evita recalcular o IDF (Inverse Document Frequency) para cada busca,
    // tornando as consultas muito mais rápidas.
    doc_frequencies: AHashMap<TermId, usize>,
}

impl ShardedIndex {
    /// Cria um novo `ShardedIndex` vazio com um número especificado de shards.
    pub fn new(num_shards: usize) -> Self {
        let shards = (0..num_shards)
            .map(|_| IndexShard {
                postings: AHashMap::default(),
                docs: AHashMap::default(),
                total_doc_len: 0,
                avg_doc_len: 0.0,
            })
            .collect();

        ShardedIndex {
            shards,
            file_table: Vec::new(),
            stop_words: Self::get_stop_words(),
            total_docs: 0,
            term_dictionary: TermDictionary::default(),
            doc_frequencies: AHashMap::default(), // OTIMIZAÇÃO: Inicializa o mapa de frequências.
        }
    }

    /// Indexa todos os arquivos `.txt` em um diretório especificado.
    pub fn index_directory(&mut self, dir_path: &Path) -> io::Result<()> {
        info!("Buscando arquivos .txt no diretório: {:?}", dir_path);
        let files: Vec<_> = fs::read_dir(dir_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt"))
            .collect();

        self.file_table = files;
        info!("Encontrados {} arquivos para indexar.", self.file_table.len());

        let pb = ProgressBar::new(self.file_table.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let doc_id_counter = AtomicUsize::new(0);
        // Estrutura de dados temporária para coletar dados da indexação em paralelo.
        // Cada thread trabalhará em sua própria entrada de `shard_data` para evitar contenção de locks.
        let shard_data: Vec<_> = (0..self.shards.len())
            .map(|_| {
                Mutex::new((
                    AHashMap::<TermId, Vec<TermPosting>>::default(), // Postings
                    AHashMap::<usize, DocumentMetadata>::default(),  // Docs
                    0u64,                                            // Total doc len
                ))
            })
            .collect();
        
        let term_dictionary = Mutex::new(TermDictionary::default());

        for (file_id, path) in self.file_table.iter().enumerate() {
            debug!("Processando arquivo: {:?}", path);
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);

                // Processa o arquivo em blocos (chunks) para melhor paralelismo e uso de memória.
                for (chunk_idx, chunk) in reader.lines().chunks(CHUNK_SIZE).into_iter().enumerate() {
                    let lines_chunk: Vec<String> = chunk.filter_map(Result::ok).collect();
                    let base_line_num = chunk_idx * CHUNK_SIZE;

                    // O coração do processamento paralelo: cada linha do chunk é processada em uma thread separada (pelo Rayon).
                    lines_chunk
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(i, line)| {
                            let doc_id = doc_id_counter.fetch_add(1, Ordering::Relaxed);
                            let shard_index = doc_id % self.shards.len(); // Distribui documentos entre as shards.
                            let line_number = base_line_num + i + 1;

                            // Usa o stemmer local da thread.
                            STEMMER.with(|stemmer_cell| {
                                let stemmer = stemmer_cell.borrow();
                                let tokens = self.tokenize_and_stem(&line, &stemmer);
                                let doc_len = tokens.len() as u32;

                                // Coleta as posições de cada termo no documento atual.
                                let mut term_positions: AHashMap<TermId, Vec<u32>> = AHashMap::default();
                                for (pos, token) in tokens.into_iter().enumerate() {
                                    // Bloqueia o dicionário, insere o termo e obtém o ID.
                                    let term_id = {
                                        let mut dict = term_dictionary.lock().unwrap();
                                        dict.get_or_insert(&token)
                                    };
                                    term_positions.entry(term_id).or_default().push(pos as u32);
                                }

                                // Bloqueia os dados da shard correspondente e insere as informações do documento.
                                let mut shard_lock = shard_data[shard_index].lock().unwrap();
                                let (ref mut postings, ref mut docs, ref mut total_doc_len) = *shard_lock;

                                *total_doc_len += doc_len as u64;
                                docs.insert(doc_id, DocumentMetadata { file_id, line_number, doc_len });

                                for (term_id, positions) in term_positions {
                                    postings
                                        .entry(term_id)
                                        .or_default()
                                        .push(TermPosting { doc_id, positions });
                                }
                            });
                        });
                }
            } else {
                warn!("Não foi possível abrir o arquivo: {:?}", path);
            }
            pb.inc(1);
        }
        pb.finish_with_message("Finalizando a indexação...");

        // Mescla os dados temporários das shards e o dicionário na estrutura principal do índice.
        self.term_dictionary = term_dictionary.into_inner().unwrap();
        self.total_docs = doc_id_counter.load(Ordering::Relaxed);

        info!("Mesclando dados das shards...");
        for (i, shard_lock) in shard_data.into_iter().enumerate() {
            let (postings, docs, total_doc_len) = shard_lock.into_inner().unwrap();
            let num_docs = docs.len();
            self.shards[i].postings = postings;
            self.shards[i].docs = docs;
            self.shards[i].total_doc_len = total_doc_len;
            self.shards[i].avg_doc_len = if num_docs > 0 {
                (total_doc_len as f64) / (num_docs as f64)
            } else {
                0.0
            };
        }

        // OTIMIZAÇÃO PÓS-INDEXAÇÃO: Pré-calcula frequências globais e ordena as listas de postings.
        // Este é um custo único na indexação que acelera muito as buscas.
        info!("Calculando frequências de documentos e ordenando listas de postings...");
        let mut doc_frequencies = AHashMap::default();
        for shard in &mut self.shards {
             shard.postings.par_iter_mut().for_each(|(&_term_id, postings)| {
                // Ordenar por doc_id permite buscas binárias rápidas durante a consulta.
                postings.sort_unstable_by_key(|p| p.doc_id);
                // A agregação de frequências precisa ser feita após o loop paralelo.
            });

             for (&term_id, postings) in &shard.postings {
                 *doc_frequencies.entry(term_id).or_insert(0) += postings.len();
             }
        }
        self.doc_frequencies = doc_frequencies;
        info!("Otimização do índice concluída.");

        Ok(())
    }

    /// Realiza uma busca no índice com base em uma string de consulta.
    pub fn search(&self, query: &str) -> Vec<(PathBuf, usize, f64)> {
        let stemmer = Stemmer::create(Algorithm::English);
        let query_token_strings = self.tokenize_and_stem(query, &stemmer);

        // Converte os tokens da consulta para seus TermIds, removendo duplicatas e tokens não encontrados.
        let query_token_ids: Vec<TermId> = query_token_strings
            .iter()
            .filter_map(|token| self.term_dictionary.map.get(token).copied())
            .collect::<HashSet<_>>() // Garante tokens únicos
            .into_iter()
            .collect();

        if query_token_ids.is_empty() {
            info!("Nenhum termo da consulta foi encontrado no dicionário.");
            return Vec::new();
        }
        debug!("IDs dos termos da consulta: {:?}", query_token_ids);

        // OTIMIZAÇÃO: Pré-calcula o IDF para cada termo da consulta uma única vez.
        let idf_map: AHashMap<TermId, f64> = query_token_ids
            .par_iter()
            .map(|&term_id| {
                let doc_freq = self.doc_frequencies.get(&term_id).copied().unwrap_or(0);
                (term_id, self.calculate_idf(doc_freq))
            })
            .collect();

        // Executa a busca em paralelo em todas as shards.
        let all_results: Vec<_> = self.shards
            .par_iter()
            .flat_map(|shard| {
                let mut doc_scores: AHashMap<usize, f64> = AHashMap::default();

                // 1. Calcula as pontuações BM25 para cada documento relevante.
                for &token_id in &query_token_ids {
                    if let (Some(postings), Some(idf)) = (shard.postings.get(&token_id), idf_map.get(&token_id)) {
                        for posting in postings {
                            let score = self.calculate_bm25(posting, *idf, shard);
                            *doc_scores.entry(posting.doc_id).or_insert(0.0) += score;
                        }
                    }
                }

                // 2. Para cada documento pontuado, calcula o "boost" de proximidade e combina as pontuações.
                doc_scores
                    .into_iter()
                    .map(|(doc_id, bm25_score)| {
                        let proximity_boost = self.calculate_proximity_score(doc_id, &query_token_ids, shard);
                        let metadata = &shard.docs[&doc_id];
                        let file_path = self.file_table[metadata.file_id].clone();
                        // Pontuação final = BM25 * (1 + Boost de Proximidade)
                        let final_score = bm25_score * (1.0 + proximity_boost);
                        (file_path, metadata.line_number, final_score)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut final_results = all_results;
        // OTIMIZAÇÃO: Usa uma ordenação paralela para grandes conjuntos de resultados.
        final_results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        final_results
    }
    
    /// Calcula a pontuação BM25 para um termo em um documento.
    fn calculate_bm25(&self, posting: &TermPosting, idf: f64, shard: &IndexShard) -> f64 {
        let doc_len = shard.docs[&posting.doc_id].doc_len as f64;
        let tf = posting.positions.len() as f64; // Term Frequency
        let numerator = tf * (K1 + 1.0);
        let denominator = tf + K1 * (1.0 - B + B * (doc_len / shard.avg_doc_len));
        idf * (numerator / denominator)
    }

    /// Calcula o IDF (Inverse Document Frequency) usando a frequência global pré-calculada.
    fn calculate_idf(&self, doc_freq: usize) -> f64 {
        let total_docs = self.total_docs as f64;
        // Usa a variante BM25+ do IDF, que é mais estável.
        ((total_docs - (doc_freq as f64) + 0.5) / ((doc_freq as f64) + 0.5) + 1.0).ln()
    }
    
    /// Calcula uma pontuação de proximidade baseada na menor distância entre os termos da consulta em um documento.
    /// Documentos onde os termos da consulta aparecem próximos recebem uma pontuação maior.
    fn calculate_proximity_score(&self, doc_id: usize, query_token_ids: &[TermId], shard: &IndexShard) -> f64 {
        if query_token_ids.len() < 2 {
            return 0.0; // Não há proximidade a ser calculada com menos de dois termos.
        }

        let mut positions_with_terms: Vec<(u32, TermId)> = Vec::new();
        for &token_id in query_token_ids {
            if let Some(postings) = shard.postings.get(&token_id) {
                // OTIMIZAÇÃO: Usa busca binária nas listas de postings pré-ordenadas para encontrar o doc_id rapidamente.
                if let Ok(idx) = postings.binary_search_by_key(&doc_id, |p| p.doc_id) {
                    let posting = &postings[idx];
                    positions_with_terms.extend(posting.positions.iter().map(|&pos| (pos, token_id)));
                }
            }
        }

        if positions_with_terms.len() < 2 {
            return 0.0;
        }

        // Ordena todos os termos encontrados pela sua posição no documento.
        positions_with_terms.sort_unstable_by_key(|k| k.0);
        
        // Encontra a distância mínima entre duas janelas de termos *diferentes*.
        let min_dist = positions_with_terms
            .windows(2)
            .filter_map(|w| {
                if w[0].1 != w[1].1 { // Compara por TermId para garantir que são termos diferentes.
                    Some(w[1].0 - w[0].0)
                } else {
                    None
                }
            })
            .min()
            .unwrap_or(u32::MAX);

        if min_dist == u32::MAX {
            0.0 // Nenhum par de termos diferentes foi encontrado.
        } else {
            // A pontuação é inversamente proporcional à distância.
            1.0 / ((min_dist as f64) + 1.0)
        }
    }

    /// Processa uma string de texto: tokeniza, remove pontuação, converte para minúsculas, remove stop words e aplica stemming.
    fn tokenize_and_stem<'a>(&self, text: &'a str, stemmer: &'a Stemmer) -> Vec<String> {
        TOKENIZER_REGEX.split(text)
            .map(|s| s.trim_matches(|p: char| !p.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty() && !self.stop_words.contains(s))
            .map(|s| stemmer.stem(&s).into_owned())
            .collect()
    }

    /// Retorna um HashSet com uma lista de stop words em inglês.
    fn get_stop_words() -> HashSet<String> {
        [
            "a", "an", "the", "in", "on", "of", "for", "to", "with", "is", "are", "was",
            "were", "at", "by", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "should", "can", "could", "may", "might",
            "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
            "these", "those", "am",
        ].iter().map(|s| s.to_string()).collect()
    }

    /// Salva o estado atual do índice em um arquivo, usando compressão zstd.
    fn save(&self, index_path: &Path) -> io::Result<()> {
        let file = File::create(index_path)?;
        let buffered_writer = BufWriter::new(file);
        // Nível de compressão 3 é um bom equilíbrio entre velocidade e taxa de compressão.
        let mut encoder = Encoder::new(buffered_writer, 3)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        bincode::serialize_into(&mut encoder, self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        encoder.finish()?;
        Ok(())
    }

    /// Carrega um índice de um arquivo, descomprimindo com zstd.
    fn load(index_path: &Path) -> io::Result<Self> {
        let file = File::open(index_path)?;
        let buffered_reader = BufReader::new(file);
        let decoder = Decoder::new(buffered_reader)?;
        bincode::deserialize_from(decoder)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

/// Função auxiliar para ler uma linha específica de um arquivo.
fn get_line_from_file(file_path: &Path, line_number: usize) -> io::Result<Option<String>> {
    if line_number == 0 {
        return Ok(None);
    }
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    // `.nth()` é O(n), mas para exibir resultados é aceitável.
    Ok(reader.lines().nth(line_number - 1).transpose()?)
}

// ==============================
//  Lógica Principal da Aplicação (CLI)
// ==============================

/// Mycelium: Uma ferramenta de busca de texto completo em Rust, rápida e multithreaded.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, help_template = "\
{name} {version}
{author-with-newline}
{about-with-newline}
{usage-heading} {usage}

{all-args}
")]
struct Cli {
    /// Aumenta o nível de verbosidade. Use -v para INFO, -vv para DEBUG.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Indexa um diretório de arquivos de texto (.txt).
    Index {
        /// O caminho para o diretório a ser indexado.
        #[arg(value_name = "DIRETÓRIO")]
        dir_path: PathBuf,

        /// [Opcional] O caminho para salvar o arquivo de índice.
        #[arg(short, long, value_name = "ARQUIVO_DE_ÍNDICE", default_value = "index.bin")]
        output: PathBuf,
    },
    /// Procura por uma consulta em um arquivo de índice existente.
    Search {
        /// A consulta de pesquisa a ser executada.
        #[arg(value_name = "CONSULTA")]
        query: String,

        /// [Opcional] O caminho para o arquivo de índice a ser usado.
        #[arg(short, long, value_name = "ARQUIVO_DE_ÍNDICE", default_value = "index.bin")]
        index_file: PathBuf,

        /// [Opcional] Salva os resultados em um arquivo em vez de imprimir no console.
        #[arg(short, long, value_name = "ARQUIVO_DE_SAÍDA")]
        output: Option<PathBuf>,
    },
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    // Configura o logger com base na flag de verbosidade -v.
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,  // Padrão: mostra apenas avisos e erros.
        1 => log::LevelFilter::Info,  // -v: mostra informações gerais.
        _ => log::LevelFilter::Debug, // -vv, -vvv, etc.: mostra informações de depuração.
    };
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_secs()
        .init();

    match cli.command {
        Commands::Index { dir_path, output } => {
            let num_shards = num_cpus::get();
            info!("Iniciando indexação com {} shards (núcleos de CPU)...", num_shards);
            let start = Instant::now();
            let mut index = ShardedIndex::new(num_shards);
            index.index_directory(&dir_path)?;
            info!("Processamento dos dados de indexação concluído em {:?}.", start.elapsed());

            let save_start = Instant::now();
            index.save(&output)?;
            let final_size = fs::metadata(&output)?.len();
            println!(
                "Índice salvo em {:?} (Tamanho: {:.2} MB) em {:?}.",
                output,
                (final_size as f64) / 1_048_576.0, // Bytes para Megabytes
                save_start.elapsed()
            );
        }
        Commands::Search { query, index_file, output } => {
            if !index_file.exists() {
                eprintln!(
                    "Erro: Arquivo de índice não encontrado em {:?}. Por favor, execute o comando 'index' primeiro.",
                    index_file
                );
                return Ok(());
            }

            info!("Carregando índice de {:?}...", index_file);
            let load_start = Instant::now();
            let index = ShardedIndex::load(&index_file)?;
            info!("Índice carregado em {:?}.", load_start.elapsed());

            info!("Buscando por: '{}'", query);
            let search_start = Instant::now();
            let results = index.search(&query);
            let search_duration = search_start.elapsed();
            
            // Se um caminho de saída foi fornecido, salva os resultados em um arquivo.
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
                    let line_content = get_line_from_file(file_path, *line_number)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| "[Não foi possível ler a linha]".to_string());
                    
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
                // Comportamento padrão: imprime os melhores resultados no console.
                if !results.is_empty() {
                    println!("\nEncontrados {} resultados em {:?}:", results.len(), search_duration);
                    // Mostra os 10 melhores resultados.
                    for (file_path, line_number, score) in results.iter().take(10) {
                        let line_content = get_line_from_file(file_path, *line_number)
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| "[Não foi possível ler a linha]".to_string());
                        
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