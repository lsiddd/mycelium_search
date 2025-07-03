use indicatif::{ ProgressBar, ProgressStyle };
use itertools::Itertools;
use log::info;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut; // OPTIMIZATION: For parallel sorting
use regex::Regex;
use rust_stemmers::{ Algorithm, Stemmer };
use serde::{ Deserialize, Serialize };
use std::cell::RefCell;
use std::collections::{ HashMap, HashSet };
use std::env;
use std::fs::{ self, File };
use std::hash::BuildHasherDefault;
use std::io::{ self, BufRead, BufReader, BufWriter, Write }; // FEATURE: Added `Write` for saving results
use std::path::{ Path, PathBuf };
use std::sync::atomic::{ AtomicUsize, Ordering };
use std::sync::Mutex;
use std::time::Instant;
use zstd::stream::{ Decoder, Encoder };

// ============================== Optimizations & Type Aliases ==============================

// OPTIMIZATION: Use a much faster hashing algorithm (AHash) for all HashMaps.
type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<ahash::AHasher>>;

// ============================== Constants & Statics ==============================
const K1: f64 = 1.5;
const B: f64 = 0.75;
const CHUNK_SIZE: usize = 20_000;

static TOKENIZER_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s:|/]+").unwrap());
thread_local! {
    static STEMMER: RefCell<Stemmer> = RefCell::new(Stemmer::create(Algorithm::English));
}

// ============================== Data Structures ==============================
type TermId = u32;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TermPosting {
    doc_id: usize,
    positions: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DocumentMetadata {
    file_id: usize,
    line_number: usize,
    doc_len: u32,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct TermDictionary {
    map: AHashMap<String, TermId>, // OPTIMIZATION: Use faster hasher
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

#[derive(Debug, Serialize, Deserialize)]
struct IndexShard {
    postings: AHashMap<TermId, Vec<TermPosting>>, // OPTIMIZATION: Use faster hasher
    docs: AHashMap<usize, DocumentMetadata>, // OPTIMIZATION: Use faster hasher
    total_doc_len: u64,
    avg_doc_len: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShardedIndex {
    shards: Vec<IndexShard>,
    file_table: Vec<PathBuf>,
    stop_words: HashSet<String>,
    total_docs: usize,
    term_dictionary: TermDictionary,
    // OPTIMIZATION: Store global document frequencies for fast IDF calculation.
    doc_frequencies: AHashMap<TermId, usize>,
}

impl ShardedIndex {
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
            doc_frequencies: AHashMap::default(), // OPTIMIZATION: Initialize doc frequencies map.
        }
    }

    pub fn index_directory(&mut self, dir_path: &Path) -> io::Result<()> {
        let files: Vec<_> = fs
            ::read_dir(dir_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(
                |path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt")
            )
            .collect();

        self.file_table = files;

        let pb = ProgressBar::new(self.file_table.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
                )
                .unwrap()
                .progress_chars("#>-")
        );

        let doc_id_counter = AtomicUsize::new(0);
        // OPTIMIZATION: Use faster AHashMap for temporary shard data.
        // OPTIMIZATION: Use faster AHashMap for temporary shard data.
        // By explicitly defining the types here, we resolve the compiler's type inference ambiguity.
        let shard_data: Vec<_> = (0..self.shards.len())
            .map(|_| {
                Mutex::new((
                    AHashMap::<TermId, Vec<TermPosting>>::default(), // Explicitly typed
                    AHashMap::<usize, DocumentMetadata>::default(), // Explicitly typed
                    0u64,
                ))
            })
            .collect();

        let term_dictionary = Mutex::new(TermDictionary::default());

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
                            let doc_id = doc_id_counter.fetch_add(1, Ordering::Relaxed);
                            let shard_index = doc_id % self.shards.len();
                            let line_number = base_line_num + i + 1;

                            STEMMER.with(|stemmer_cell| {
                                let stemmer = stemmer_cell.borrow();
                                let tokens = self.tokenize_and_stem(&line, &stemmer);
                                let doc_len = tokens.len() as u32;

                                let mut term_positions: AHashMap<
                                    TermId,
                                    Vec<u32>
                                > = AHashMap::default();
                                for (pos, token) in tokens.into_iter().enumerate() {
                                    let term_id = {
                                        let mut dict = term_dictionary.lock().unwrap();
                                        dict.get_or_insert(&token)
                                    };
                                    term_positions
                                        .entry(term_id)
                                        .or_default()
                                        .push(pos as u32);
                                }

                                let mut shard_lock = shard_data[shard_index].lock().unwrap();
                                let (ref mut postings, ref mut docs, ref mut total_doc_len) =
                                    *shard_lock;

                                *total_doc_len += doc_len as u64;
                                docs.insert(doc_id, DocumentMetadata {
                                    file_id,
                                    line_number,
                                    doc_len,
                                });

                                for (term_id, positions) in term_positions {
                                    postings
                                        .entry(term_id)
                                        .or_default()
                                        .push(TermPosting { doc_id, positions });
                                }
                            });
                        });
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Finalizing index...");

        self.term_dictionary = term_dictionary.into_inner().unwrap();
        self.total_docs = doc_id_counter.load(Ordering::Relaxed);

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

        // OPTIMIZATION: Pre-calculate global document frequencies and sort posting lists.
        // This is a one-time cost at indexing that makes searches much faster.
        let mut doc_frequencies = AHashMap::default();
        for shard in &mut self.shards {
            for (&term_id, postings) in &mut shard.postings {
                // Sorting allows for fast binary searches during query time.
                postings.sort_unstable_by_key(|p| p.doc_id);
                *doc_frequencies.entry(term_id).or_insert(0) += postings.len();
            }
        }
        self.doc_frequencies = doc_frequencies;

        Ok(())
    }

    /// Performs a search query against the index.
    pub fn search(&self, query: &str) -> Vec<(PathBuf, usize, f64)> {
        let stemmer = Stemmer::create(Algorithm::English);
        let query_token_strings = self.tokenize_and_stem(query, &stemmer);

        let query_token_ids: Vec<TermId> = query_token_strings
            .iter()
            .filter_map(|token| self.term_dictionary.map.get(token).copied())
            .collect::<HashSet<_>>() // Ensure unique tokens
            .into_iter()
            .collect();

        if query_token_ids.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION: Pre-calculate IDF for each query term once, before any searching.
        let idf_map: AHashMap<TermId, f64> = query_token_ids
            .par_iter()
            .map(|&term_id| {
                let doc_freq = self.doc_frequencies.get(&term_id).copied().unwrap_or(0);
                (term_id, self.calculate_idf(doc_freq))
            })
            .collect();

        let all_results: Vec<_> = self.shards
            .par_iter()
            .flat_map(|shard| {
                let mut doc_scores: AHashMap<usize, f64> = AHashMap::default();

                // Calculate BM25 scores
                for &token_id in &query_token_ids {
                    if
                        let (Some(postings), Some(idf)) = (
                            shard.postings.get(&token_id),
                            idf_map.get(&token_id),
                        )
                    {
                        for posting in postings {
                            let score = self.calculate_bm25(posting, *idf, shard);
                            *doc_scores.entry(posting.doc_id).or_insert(0.0) += score;
                        }
                    }
                }

                // Calculate proximity boost and combine scores
                doc_scores
                    .into_iter()
                    .map(|(doc_id, bm25_score)| {
                        let proximity_boost = self.calculate_proximity_score(
                            doc_id,
                            &query_token_ids,
                            shard
                        );
                        let metadata = &shard.docs[&doc_id];
                        let file_path = self.file_table[metadata.file_id].clone();
                        (file_path, metadata.line_number, bm25_score * (1.0 + proximity_boost))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut final_results = all_results;
        // OPTIMIZATION: Use a parallel sort for large result sets.
        final_results.par_sort_unstable_by(|a, b|
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        );
        final_results
    }

    fn calculate_bm25(&self, posting: &TermPosting, idf: f64, shard: &IndexShard) -> f64 {
        let doc_len = shard.docs[&posting.doc_id].doc_len as f64;
        let tf = posting.positions.len() as f64;
        let numerator = tf * (K1 + 1.0);
        let denominator = tf + K1 * (1.0 - B + B * (doc_len / shard.avg_doc_len));
        idf * (numerator / denominator)
    }

    /// Calculates IDF using the pre-computed global document frequency.
    fn calculate_idf(&self, doc_freq: usize) -> f64 {
        let total_docs = self.total_docs as f64;
        // Use the BM25+ variant of IDF
        ((total_docs - (doc_freq as f64) + 0.5) / ((doc_freq as f64) + 0.5) + 1.0).ln()
    }

    /// Greatly optimized proximity score calculation.
    fn calculate_proximity_score(
        &self,
        doc_id: usize,
        query_token_ids: &[TermId],
        shard: &IndexShard
    ) -> f64 {
        if query_token_ids.len() < 2 {
            return 0.0;
        }

        let mut positions_with_terms: Vec<(u32, TermId)> = Vec::new();
        for &token_id in query_token_ids {
            if let Some(postings) = shard.postings.get(&token_id) {
                // OPTIMIZATION: Use a fast binary search on the pre-sorted posting lists.
                if let Ok(idx) = postings.binary_search_by_key(&doc_id, |p| p.doc_id) {
                    let posting = &postings[idx];
                    positions_with_terms.extend(
                        posting.positions.iter().map(|&pos| (pos, token_id))
                    );
                }
            }
        }

        if positions_with_terms.len() < 2 {
            return 0.0;
        }

        positions_with_terms.sort_unstable_by_key(|k| k.0);

        let min_dist = positions_with_terms
            .windows(2)
            .filter_map(|w| {
                if w[0].1 != w[1].1 {
                    // Compare by TermId
                    Some(w[1].0 - w[0].0)
                } else {
                    None
                }
            })
            .min()
            .unwrap_or(u32::MAX);

        if min_dist == u32::MAX {
            0.0
        } else {
            1.0 / ((min_dist as f64) + 1.0)
        }
    }

    fn tokenize_and_stem<'a>(&self, text: &'a str, stemmer: &'a Stemmer) -> Vec<String> {
        TOKENIZER_REGEX.split(text)
            .map(|s| s.trim_matches(|p: char| !p.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty() && !self.stop_words.contains(s))
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

    fn save(&self, index_path: &Path) -> io::Result<()> {
        let file = File::create(index_path)?;
        let buffered_writer = BufWriter::new(file);
        let mut encoder = Encoder::new(buffered_writer, 3).map_err(|e|
            io::Error::new(io::ErrorKind::Other, e)
        )?;
        bincode
            ::serialize_into(&mut encoder, self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        encoder.finish()?;
        Ok(())
    }

    fn load(index_path: &Path) -> io::Result<Self> {
        let file = File::open(index_path)?;
        let buffered_reader = BufReader::new(file);
        let decoder = Decoder::new(buffered_reader)?;
        bincode::deserialize_from(decoder).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
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

// ============================== Main Application ==============================
fn main() -> io::Result<()> {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    // FEATURE: Updated usage string to include saving results.
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <index|search> <directory|query> [index_file] [output_file]",
            &args[0]
        );
        return Ok(());
    }

    let command = &args[1];
    let default_index_path = PathBuf::from("index.bin");
    let index_path = args.get(3).map(PathBuf::from).unwrap_or(default_index_path);

    match command.as_str() {
        "index" => {
            let dir_path = Path::new(&args[2]);
            let num_shards = num_cpus::get();
            info!("Starting indexing with {} shards...", num_shards);
            let start = Instant::now();
            let mut index = ShardedIndex::new(num_shards);
            index.index_directory(dir_path)?;
            info!("Indexing data processed in {:?}.", start.elapsed());

            let save_start = Instant::now();
            index.save(&index_path)?;
            let final_size = fs::metadata(&index_path)?.len();
            println!(
                "Index saved to {:?} ({:.2} MB) in {:?}.",
                index_path,
                (final_size as f64) / 1_048_576.0,
                save_start.elapsed()
            );
        }
        "search" => {
            let query = &args[2];
            // FEATURE: Get optional output file path from command line arguments.
            let output_path = args.get(4).map(PathBuf::from);

            if !index_path.exists() {
                eprintln!(
                    "Error: Index file not found at {:?}. Please run 'index' first.",
                    index_path
                );
                return Ok(());
            }

            info!("Loading index from {:?}...", index_path);
            let load_start = Instant::now();
            let index = ShardedIndex::load(&index_path)?;
            info!("Index loaded in {:?}.", load_start.elapsed());

            info!("Searching for: '{}'", query);
            let search_start = Instant::now();
            let results = index.search(query);
            let search_duration = search_start.elapsed();

            // FEATURE: Save results to file if an output path is provided.
            if let Some(out_path) = output_path {
                println!(
                    "Found {} results in {:?}. Saving to {}...",
                    results.len(),
                    search_duration,
                    out_path.display()
                );
                // NOTE: Saving to a file like 'Cargo.toml' will overwrite it. Be careful!
                let file = File::create(&out_path)?;
                let mut writer = BufWriter::new(file);

                writeln!(writer, "Search results for query: '{}'", query)?;
                writeln!(writer, "Found {} results.\n", results.len())?;

                for (file_path, line_number, score) in results.iter() {
                    let line_content = get_line_from_file(file_path, *line_number)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| "[Could not read line]".to_string());

                    writeln!(
                        writer,
                        "Score: {:.4}, File: \"{}\", Line: {}\n > {}\n",
                        score,
                        file_path.display(),
                        line_number,
                        line_content.trim()
                    )?;
                }
                println!("Results saved successfully to {}.", out_path.display());
            } else {
                // Original behavior: print top results to the console.
                if !results.is_empty() {
                    println!("\nFound {} results in {:?}:", results.len(), search_duration);
                    for (file_path, line_number, score) in results.iter().take(10) {
                        let line_content = get_line_from_file(file_path, *line_number)
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| "[Could not read line]".to_string());
                        println!(
                            "  - Score: {:.4}, File: \"{}\", Line: {}",
                            score,
                            file_path.display(),
                            line_number
                        );
                        println!("    > {}", line_content.trim());
                    }
                } else {
                    println!("\nNo results found in {:?}.", search_duration);
                }
            }
        }
        _ => {
            eprintln!("Unknown command: '{}'. Use 'index' or 'search'.", command);
        }
    }

    Ok(())
}
