# Security policy

## Supported version

Security fixes are provided for the current Domino release.

| Version | Supported |
|---|---|
| 0.3.x | Yes |

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose data, execute
untrusted code, or corrupt scientific output. Use GitHub private vulnerability
reporting if it is enabled for the repository. Otherwise contact the repository
owner privately through the account listed in `CITATION.cff`.

Include the affected version, operating system, minimal reproduction, impact,
and any proposed mitigation. Do not attach restricted genotype or phenotype
data. Synthetic reproductions are strongly preferred.

Domino reads local paths supplied by the analyst and writes output beside the
requested prefix. Run untrusted input in an appropriately isolated environment
and review output paths before launching a large analysis.
