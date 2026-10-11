# Search commands

## `search.query`

<!-- BEGIN GENERATED: search.query -->
Changes state: no.

Searches the photos of the current folder with the rules of the global search dialog. An empty text returns the recommendations. With 'mode' auto a natural language query uses the semantic path when natural language search is on; when that path is not available the result has 'semantic_unavailable' and no matches. 'mode' fuzzy always uses the text search. A query that is too long for the semantic path returns 'too_long'.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `limit` | integer | no |  | 1 to 48. The number of matches to return. Default 24. |
| `mode` | `auto` \| `fuzzy` | no |  | auto (default) or fuzzy. |
| `offset` | integer | no |  | at least 0. The first match to return. Default 0. |
| `text` | string | yes |  | The search text. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `has_more` | boolean | no |  |
| `limit` | integer | no |  |
| `matches` | array | yes |  |
| `offset` | integer | no |  |
| `recommendations` | array | no |  |
| `route` | string | yes |  |
| `semantic_error` | string | no |  |
| `semantic_unavailable` | boolean | no |  |
| `too_long` | boolean | no |  |
| `total` | integer | no |  |
<!-- END GENERATED: search.query -->

`route` tells which path answered: `traditional` (text search of file names and metadata),
`semantic` (natural language search, when it is on in Settings), or `empty` (no text; the
result has `recommendations`). Page with `offset` while `has_more` is true.

**Errors**

- `-32602` `/limit`: more than 48.
- `-32001`: `No project is open.`
- `-32002`: `A newer search replaced this search.` Only one search runs at a time.
- `-32007`: the search failed.

**Example**

```bash
alcedo-cli --json call search.query '{"text":"DSC00800"}'
```

```json
{"has_more":false,"limit":24,"matches":[{"camera_model":"ILCE-7M3","capture_date":"2023-02-18","element_id":2,"file_name":"DSC00800.ARW","image_id":2,"lens":"E 70-180mm F2.8 A056","rating":0}],"offset":0,"route":"traditional","total":1}
```

Paging, with the text search only:

```bash
alcedo-cli --json call search.query '{"text":"ILCE-7M3","mode":"fuzzy","limit":1}'
```

```json
{"has_more":true,"limit":1,"matches":[{"camera_model":"ILCE-7M3","capture_date":"2022-09-18","element_id":1,"file_name":"DSC0027.ARW","image_id":1,"lens":"FE 28mm F2","rating":4}],"offset":0,"route":"traditional","total":2}
```

The recommendations of an empty text:

```bash
alcedo-cli --json call search.query '{"text":""}'
```

```json
{"matches":[],"recommendations":[{"category":"camera","categoryLabel":"Camera","count":2,"label":"ILCE-7M3","query":"ILCE-7M3"},{"category":"date","categoryLabel":"Date","count":1,"label":"2023-10-22","query":"2023-10-22"},{"category":"lens","categoryLabel":"Lens","count":1,"label":"FE 28mm F2","query":"FE 28mm F2"}, ...],"route":"empty"}
```
