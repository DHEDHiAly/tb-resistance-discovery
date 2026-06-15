CRyPTIC REUSE README


This folder contains data recommended for most users of the CRyPTIC data. It
has all the information necessary to use the data in your own study. Please see
the reproducibility folder if instead you need to reproduce any of the CRyPTIC
publications.

If you publish using this data, please cite our compendium paper, currently a
preprint: The CRyPTIC Consortium, “A data compendium of Mycobacterium
tuberculosis antibiotic resistance”, https://doi.org/10.1101/2021.09.14.460274.

This folder contains one file: CRyPTIC_reuse_table_20221019.csv. This is the
main file of sample data. For each sample it has the ENA sample ID, phenotype data
(binary classifications (see https://www.medrxiv.org/content/10.1101/2021.02.24.21252386v1.full.pdf
for more) and minimum inhibitory concentrations (MIC, relative to CRyPTIC
designed 96-well plate (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6125532/
for more), and names of VCF files. The columns are described below.

The file CRyPTIC_reuse_table_20221019.csv has the following columns.

    * UNIQUEID. This is a unique identifier that is used elsewhere for
      reproducibility. It is in this file for completeness and reproducibility.
      We recommend that the ENA sample identifier (in the ENA_SAMPLE) column is
      used to track samples, for readability.

    * XYZ_BINARY_PHENOTYPE. These columns contain the binary phenotype call for
      each drug. The drugs are named using their 3 letter codes: AMI
      (Amikacin), BDQ (Bedaquiline), CFZ (Clofazimine), DLM (Delamanid), EMB
      (Ethambutol), ETH (Ethionamide), INH (Isoniazid), KAN (Kanamycin), LEV
      (Levofloxacin), LZD (Linezolid), MXF (Moxifloxacin), RIF (Rifampicin),
      RFB (Rifabutin).  The epidemiological cutoff values (ECOFFs) used to
      determined these values are listed in the table at the end of this file.

    * XYZ_MIC. These columns have the MIC value for each drug (using the same 3
      letter codes as for the binary phenotype)

    * XYZ_PHENOTYPE_QUALITY. These columns contain the quality of each
      phenotype call. Phenotypes were measured using three different methods:
      1) by a scientist in the laboratory using images of the 96-well assay
      plates taken with the Thermo Fisher Sensititre™Vizion™ digital MIC
      viewing system; 2) by the Automated Mycobacterial Growth Detection
      Algorithm (AMyGDA) software; 3) by ≥11 volunteers as part of the citizen
      science project, BashTheBug. Please see the Compendium paper for further
      details on these methods. MICs were then classified as “HIGH” (at least
      two methods concur on the MIC), “MEDIUM” (either there is no plate image,
      or Vizion and AMyGDA disagree and there is no BashTheBug measurement), or
      “LOW” (all three methods disagree) quality. 

    * ENA_SAMPLE. The ENA sample accession. This allows you to look up the
      sample in the ENA (https://www.ebi.ac.uk/ena/browser/home), and then
      download the reads FASTQ files.

    * VCF. The path to the VCF file of variant calls for this sample
      (containing records only where the sample differs from the reference).
      These are calls made independently of all other samples, in contrast to
      the next column JOINT_GENOTYPED_VCF.

    * JOINT_GENOTYPED_VCF. This is the result after joint genotype calling all
      samples, which entailed using all variants from all samples in the
      per-sample VCF files (as in the previous VCF column), and genotyping
      every sample again at all those sites. Thus the joint genotyped VCF file
      for a sample contains genotype calls at all known variant sites. The
      sites in every VCF file are identical, so that samples can be
      consistently compared. The joint genotyping methods are described in
      detail in the CRyPTIC publication “Minos: variant adjudication and joint
      genotyping of cohorts of bacterial genomes” DOI
      https://doi.org/10.1101/2021.09.15.460475.


Table of epidemiological cutoff values (ECOFFs) used for each drug to determine
for a given MIC value, whether the phenotype is Susceptible (S), Intermediate
(I), or Resistant (S). This is a copy of Table 2 from
https://www.medrxiv.org/content/10.1101/2021.02.24.21252386v1.full.pdf.

Drug                ECOFF/ECV(mg/L)  Susceptible(mg/L)  Intermediate(mg/L)  Resistant(mg/L)
Isoniazid (INH)     0.1              <=0.1              0.2,0.4             >=0.8
Rifampicin (RIF)    0.5              <=0.5              -                   >=1.0
Ethambutol (EMB)    4                <=2                4                   >=8
Moxifloxacin (MXF)  1                <=1                -                   >=2
Levofloxacin (LEV)  1                <=1                -                   >=2
Kanamycin (KAN)     4                <=4                -                   >=8
Amikacin (AMI)      1                <=1                -                   >=2
Ethionamide (ETH)   4                <=2                4                   >=8
Rifabutin (RFB)     0.12             <=0.12             -                   >=0.25
Clofazimine (CFZ)   0.25             <=0.25             -                   >=0.5
Linezolid (LZD)     1                <=1                -                   >=2
Delamanid (DLM)     0.12             <=0.12             -                   >=0.25
Bedaquiline (BDQ)   0.25             <=0.25             -                   >=0.5
