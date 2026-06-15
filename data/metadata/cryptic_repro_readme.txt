CRyPTIC REPRODUCIBILITY README


Using the CRyPTIC sample index

The genomic data index, cryptic-index_February2022.json, is serialised as a JSON
document. Records are indexed by unique identifiers and paths are relative to
the parent directory of the JSON document:

"site.42.iso.2.subject.XYZ.lab_id.ABC.seq_reps.1": {
    "filename": "00/05/43/21/54321/site.42.iso.2.subject.XYZ.lab_id.ABC.seq_reps.1.masked.vcf.gz",
    "lab_id": "ABC",
    "site": "42",
    "subject": 54321,
    "vcf_sample": "site.42.iso.2.subject.XYZ.lab_id.ABC.seq_reps.1",
    "iso": "2",
    "seq_reps": {,
        "rep1": {
            "r1 md5": "1230d160338fe9050aaae1d11935ea7c",
            "r2 md5": "567b11d978c8a5de963b13bc769f6ddd",
            "ENA run": "EEE7654321"
        }
    },
    "ENA": "AAA1234567",
    "GPI": false,
    "brankin-malone-2021": false,
    "regenotyped_vcf": "00/05/43/21/54321/site.42.iso.2.subject.XYZ.lab_id.ABC.seq_reps.1.regeno.vcf.gz",
    "plate_md5": "123456f5b5d9375420e2cd1ca0abcdef",
    "plate_image": "00/05/43/21/54321/42-XYZ-ABC-2-14-UKMYC5-raw.png"
},

Additional information about the ENA submission status and membership in
certain datasets is also recorded. The genotype-phenotype intersection ("GPI")
is the set of samples with both variant call data and plate images. 

For more information about the curated set of "brankin-malone-2021" samples
please see the reuse/ directory.

Samples names have five parts:

    * Site
    * Subject
    * Lab id
    * Isolate
    * Replicate

There are three contexts where different combinations of these components
will refer to a sample's data:

    * Genomic analysis, which may have combined multiple sequencing replicates
    * Phenotype data, which does not regard the sequencing replicate number
    * Raw sequencing data, which is uniquely indexed by all five properties 

Phenotype data:

Phenotype data releases are organised in the data_tables directory. Phenotype
data releases index samples in the format: site.42.subj.XYZ.lab.ABC.iso.2

Note that there are no sequencing replicates associated with phenotype data.
