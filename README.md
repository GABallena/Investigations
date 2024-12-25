# IMG/PR data

## Files

The following files are available for download:

* IMGPR_plasmid_data.tsv: Metadata related to plasmids in IMG/PR.
* IMGPR_nucl.fna.gz: Nucleotide sequences of plasmids in IMG/PR.
* IMGPR_prot.faa.gz: Sequences of proteins encoded by plasmids in IMG/PR.

## Plasmid data

The `IMGPR_plasmid_data.tsv` file contains the following columns, which describe metadata associated with each plasmid within IMG/PR:

1. plasmid_id: Unique identifier of the plasmid sequence.
2. ptu: PTU to which the plasmid was assigned.
3. taxon_oid: Identifier of the dataset IMG/M in which the plasmid was identified.
4. scaffold_oid: Identifier of the IMG/M scaffold associated with the plasmid.
5. source_type: Type of sample in which the plasmid was identified: isolate genome, MAG, SAG, metagenome, or metatranscriptome.
6. ecosystem: Ecosystem classification of the sample in which the plasmid was identified, adhering to the GOLD Ecosystem Classification schema of the Genomes OnLine Database.
7. length: Length of the plasmid nucleotide sequence.
8. gene_count: Number of genes encoded by the plasmid sequence.
9. genomad_score: geNomad plasmid score of the sequence, indicating the degree of confidence that the sequence represents a plasmid.
10. putatively_complete: Indication of whether the sequence represents a complete plasmid, spanning from one end to the other [^1].
11. topology: Topology of the plasmid sequence, indicating whether it contains direct terminal repeats (DTRs), inverted terminal repeats (ITRs), concatemers, or is linear [^2].
12. mob_genes: Relaxase (MOB) genes encoded by the plasmid, with identifiers sourced from the CONJscan HMM models.
13. t4cp_genes: Type IV coupling protein (T4CP) genes encoded by the plasmid, with identifiers sourced from the CONJscan HMM models.
14. t4ss_atpase_genes: Type IV secretion system (T4SS) ATPase genes encoded by the plasmid, with identifiers sourced from the CONJscan HMM models.
15. other_conjugation_genes: Other genes of the conjugation machinery, with identifiers sourced from the CONJscan HMM models.
16. complete_mpf_family: Whether the sequence encodes all the necessary genes for autonomous conjugation, as per CONJscan criteria. The listed identifiers correspond to the MPF types for which the criteria were met.
17. origin_of_transfer: Origin of transfer detected within the plasmid nucleotide sequence, with accessions sourced from Ares-Arroyo, M. et al. (2023) [^3].
18. arg_genes: Antibiotic resistance genes encoded by the plasmid, with identifiers sourced from the Resfams database.
19. putative_phage_plasmid: Whether the sequence encodes a virus hallmark protein and was, consequently, classified as a phage-plasmid.
20. host_prediction_method: Methodology used for host assignment (isolate taxonomy or CRISPR spacer match).
21. host_taxonomy: Taxonomic lineage of the inferred host organism.
22. closest_reference: The reference plasmid that best aligned with the sequence [^4].
23. closest_reference_ani_percent: Average nucleotide identity between the plasmid and its closest reference.
24. closest_reference_af_percent: Percentage of the plasmid's length that aligned to a reference sequence.

[^1]: Putative complete plasmids were identified through the presence of direct terminal repeats or alignment to complete reference plasmids, encompassing the entire lengths of both the query sequence and the reference, with a high Average Nucleotide Identity (ANI) (query and reference coverage ≥ 99%, ANI ≥ 95%). The reference plasmids utilized for this comparison were sourced from PLSDB (v. 2021_06_23_v2) and RefSeq (retrieved on 2022-06-16) using the query: 'archaea[filter] AND refseq[filter] AND plasmid[filter]'.

[^2]: Direct terminal repeats (DTRs) are sequences that have exactly matching parts of at least 21 bp at both ends. Inverted terminal repeats (ITRs) denote sequences in which the 5' and 3' termini are reverse complements spanning at least 21 bp. DTRs and ITRs were disregarded if more than half of their length consisted of low complexity sequences, as identified using the dustmasker tool (version 1.0.0, settings: '-level 40'). Sequences labeled as concatemers when the average frequency of canonical 21-mers was above 1.5 or when ≥1kb repeats could be identified with the repeat-match command of the MUMmer4 package (version 4.0.0rc1). Sequences lacking DTRs, ITRs, and not meeting concatemer criteria were labeled as linear.

[^3]: Ares-Arroyo, Manuel, Charles Coluzzi, and Eduardo PC Rocha. "Origins of transfer establish networks of functional dependencies for plasmid transfer by conjugation." Nucleic Acids Research 51.7 (2023): 3001-3016.

[^4]: Complete reference plasmids were obtained from PLSDB (v. 2021_06_23_v2) and RefSeq (retrieved on 2022-06-16), using the search query: 'archaea[filter] AND refseq[filter] AND plasmid[filter]'. Only references covering at least 50% of the length of the IMG/PR plasmid are listed. The closest reference was determined as the one displaying the highest value of ANI * AF (ANI: average nucleotide identity; AF: aligned fraction of the IMG/PR sequence).
