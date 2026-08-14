use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use flate2::read::GzDecoder;
use rayon::prelude::*;
use crossbeam_channel::bounded;

struct BloomFilter {
    m: usize,
    k: u32,
    bit_array: Vec<AtomicU64>,
}

impl BloomFilter {
    fn new(n_items: usize, fp_prob: f64) -> Self {
        let n = n_items as f64;
        let m = (-(n * fp_prob.ln()) / (2.0f64.ln().powi(2))).round() as usize;
        let k = (((m as f64) / n) * 2.0f64.ln()).round() as u32;

        let num_u64s = (m + 63) / 64;
        let ram_gb = (num_u64s * 8) as f64 / (1024.0 * 1024.0 * 1024.0);

        println!("Bit Array Size: {:.2} GB RAM", ram_gb);
        println!("Hash Functions (k): {}", k);

        let mut bit_array = Vec::with_capacity(num_u64s);
        for _ in 0..num_u64s {
            bit_array.push(AtomicU64::new(0));
        }

        BloomFilter { m, k, bit_array }
    }

    #[inline]
    fn add(&self, item: &[u8]) {
        let h: u64 = seahash::hash(item);

        let h1 = (h & 0xFFFFFFFF) as usize;
        let h2 = (h >> 32) as usize;

        for i in 0..self.k {
            let bit_idx = (h1.wrapping_add((i as usize).wrapping_mul(h2))) % self.m;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            let mask = 1u64 << bit_offset;

            self.bit_array[word_idx].fetch_or(mask, Ordering::Relaxed);
        }
    }

    fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        println!("\nSaving binary Bloom Filter to '{}'...", filename);
        let mut file = File::create(filename)?;

        for atomic_u64 in &self.bit_array {
            let val = atomic_u64.load(Ordering::Relaxed);
            file.write_all(&val.to_le_bytes())?;
        }

        println!("Successfully saved bloom filter!");
        Ok(())
    }
}

fn main() {
    let total_items = 1_800_000_000;
    let fp_rate = 0.0000000000001; // 0.000001% False Positive Rate
    let file_path = "addresses.txt.gz"; // Local downloaded file

    let bf = Arc::new(BloomFilter::new(total_items, fp_rate));

    println!("Opening local file '{}' and starting Async Pipeline...", file_path);
    let start_time = Instant::now();

    let (tx, rx) = bounded::<Vec<Vec<u8>>>(32);
    let bf_worker = Arc::clone(&bf);

    // WORKER THREAD: Hashing across all CPU cores
    let worker_handle = std::thread::spawn(move || {
        let mut processed_count = 0u64;
        let mut last_log = 0u64;

        for batch in rx {
            let batch_len = batch.len() as u64;
            batch.par_iter().for_each(|addr_bytes| {
                bf_worker.add(addr_bytes);
            });
            processed_count += batch_len;

            if processed_count - last_log >= 10_000_000 {
                last_log = processed_count;
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed = (processed_count as f64) / elapsed;
                println!(
                    "Processed: {} / {} ({:.2}%) | Speed: {:.0} addr/sec",
                    processed_count, total_items, (processed_count as f64 / total_items as f64) * 100.0, speed
                );
            }
        }
    });

    // PRODUCER THREAD: Fast Local Disk Gzip Reader
    let file = File::open(file_path).expect("Local dataset file 'addresses.txt.gz' not found!");
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::with_capacity(32 * 1024 * 1024, decoder); // 32MB Buffer

    let mut current_batch: Vec<Vec<u8>> = Vec::with_capacity(100_000);
    let mut line_buf = String::with_capacity(128);

    while let Ok(bytes_read) = reader.read_line(&mut line_buf) {
        if bytes_read == 0 {
            break;
        }

        let trimmed = line_buf.trim();
        if !trimmed.is_empty() {
            current_batch.push(trimmed.as_bytes().to_vec());
        }
        line_buf.clear();

        if current_batch.len() >= 100_000 {
            tx.send(current_batch).unwrap();
            current_batch = Vec::with_capacity(100_000);
        }
    }

    if !current_batch.is_empty() {
        tx.send(current_batch).unwrap();
    }

    drop(tx);
    worker_handle.join().unwrap();

    println!("\nProcessing complete!");
    bf.save_to_file("btc_addresses_bloom.bin").expect("Failed to save bloom filter binary");
}
