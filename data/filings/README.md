# SEC 10-K excerpts

Public annual reports downloaded from SEC EDGAR by [`../build_filings.py`](../build_filings.py).
Each file keeps Item 1 (Business) and Item 7 (Management's Discussion and Analysis),
converted from the filing's HTML to text; tables become `|`-separated rows.

| File | Company | Fiscal year | Accession |
| --- | --- | --- | --- |
| `uber-10k-fy2021.md` | Uber Technologies, Inc. | 2021 | 0001543151-22-000008 |
| `uber-10k-fy2024.md` | Uber Technologies, Inc. | 2024 | 0001543151-25-000008 |
| `lyft-10k-fy2024.md` | Lyft, Inc. | 2024 | 0001759509-25-000025 |
| `dash-10k-fy2024.md` | DoorDash, Inc. | 2024 | 0001628280-25-005715 |

The JSON front matter of every file records the EDGAR URL, accession number, filing date
and the SHA-256 of the downloaded document, so each excerpt can be traced to its source
and rebuilt byte for byte:

```sh
SEC_USER_AGENT="Your Name you@example.com" python data/build_filings.py
```

Excerpts only: Items 1A (Risk Factors) and 8 (Financial Statements) are not included,
which keeps the corpus small. Questions that need them will correctly find nothing.
