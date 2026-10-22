# GitHub and Zenodo release checklist

1. **License.** The software is released under the MIT License.
2. **Verify author metadata.** Confirm author name order and add ORCID identifiers only after verifying them.
3. **Run the tests locally:** `pip install -e ".[test]" && pytest -q` (expect all passing).
4. **Create the GitHub repository** (public) and push. The workflow in `.github/workflows/tests.yml` runs the tests on three Python versions.
5. **Connect Zenodo:** log in to zenodo.org with GitHub, enable the repository under *GitHub* settings.
6. **Create a GitHub release** tagged `v1.0.0`. Zenodo archives it and mints a DOI.
7. **Add the DOI** to `CITATION.cff` and README (the badge), then cite it in the manuscript's Code and Data Availability statement.
8. **Do not upload dataset images.** Upload only metadata, mapping tables, reports, and label files whose licences allow it.
9. **Freeze v1.4.** Never edit `rules_v1_4.py` after the release; add new versions as new files.
