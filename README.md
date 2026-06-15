# TB Resistance Discovery

Finding novel drug resistance mutations in *Mycobacterium tuberculosis* by scanning hundreds of genomes against the H37Rv reference and structurally validating the hits.

Part of the Kellis Lab at MIT, targeting deployment on Mantis.

---

## What's in here

### Phase 1 — Data Acquisition (done)

The pipeline currently works entirely in Python (no Linux tools needed) and runs on Windows via PowerShell. Data comes from two sources:

**NCBI:** H37Rv reference genome downloaded, plus 10 TB genome assemblies (GCF accessions) from the NCBI Assembly database. Metadata catalog for ~100 assemblies.

**CRyPTIC Consortium:** The real payload. Phenotype data for 12,287 TB clinical isolates with drug susceptibility results for 13 antibiotics. That breaks down to:
- ~4,350 MDR-TB strains (resistant to both rifampicin and isoniazid)
- ~6,500 strains resistant to at least one drug
- ~5,700 fully susceptible controls
- Every sample has a pre-computed VCF file already mapped to H37Rv

### Data structure

```
reference/H37Rv.fasta          — Reference genome (NC_000962.3)
data/genomes/*.fasta           — 10 assembled TB genomes
data/metadata/
  cryptic_phenotypes.csv       — 12,287 samples with drug resistance calls
  all_tb_metadata.csv          — 100 TB assembly records from NCBI
  all_tb_ids.json              — NCBI assembly UIDs
  tb_assemblies.csv            — Initial MDR assembly list
scripts/
  01_download_data.py          — Main download pipeline
  01b_search_more.py           — Broader NCBI search
  01c_fetch_metadata_and_download.py  — Batch metadata + genome download
  01d_check.py                 — Verify what you've got
  01e through 01l              — Exploration scripts, CRyPTIC download + analysis
```

---

## Running it

Python 3.12 with `requests`, `pandas`, and `biopython` installed. The lab network has self-signed certs so SSL verification is off in the scripts.

```powershell
cd tb-resistance-discovery
python scripts/01_download_data.py
```

Everything else in `scripts/` is exploration — run them in order to reproduce the data acquisition:

```powershell
python scripts/01b_search_more.py      # Find more TB assemblies
python scripts/01c_fetch_metadata_and_download.py  # Download a batch
python scripts/01j_download_cryptic.py  # Get CRyPTIC phenotype data
python scripts/01l_summarize.py         # See what you have
```

---

## What's next

### Phase 2 — Variant Discovery
- Align the assembled genomes against H37Rv with MUMmer or a Python-based aligner
- Or use the pre-computed CRyPTIC VCF files directly (they're sitting on EBI's FTP, just need to pull them down)
- Build a table of every SNP, insertion, and deletion across all samples

### Phase 3 — Enrichment Analysis
- Split samples by phenotype (resistant vs susceptible per drug)
- Fisher's exact test on every variant position
- Rank by enrichment in resistant strains
- Filter against known resistance mutations (the WHO catalogue, MUBII-TB-DB) to find the novel ones

### Phase 4 — Mantis Embedding
- Encode each sample's full mutation profile into Mantis's latent space
- Cluster the embeddings — resistant strains should cluster by mechanism, not just by drug
- Pull out mystery clusters: samples that are phenotypically resistant but lack known mutations

### Phase 5 — Structural Validation
- Fold mutant proteins with AlphaFold/ESMFold (RTX 4090 is perfect for this)
- Dock the drug into the mutated binding site
- Compare binding energy vs wild-type

### Phase 6 — Paper
- Methods: embedding-based resistance mutation discovery using Mantis
- Biology: the novel mutations themselves
- Target: Bioinformatics, PLOS Comp Bio, or bioRxiv preprint first

---

## Hardware

The lab machine has an RTX 4090 (24 GB VRAM) and 64 GB RAM — plenty for folding individual proteins and docking once we find candidates. The 4090 won't handle large-scale AlphaFold multimer runs but that's not what we need here.

## Notes

- Windows + bioinformatics is painful. If WSL2 gets set up at some point, `sra-tools`, `bwa`, and `samtools` become available and we can work with raw FASTQ data directly.
- The CRyPTIC dataset alone is enough for a solid publication — 12K samples with well-curated phenotypes is more than most TB resistance papers use.
- For the Alzheimer's extension later: swap the TB GWAS data for AD GWAS summary stats, swap H37Rv for the human reference, and the pipeline logic stays the same.
