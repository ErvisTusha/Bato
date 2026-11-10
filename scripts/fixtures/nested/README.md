# Nested archive fixtures

Both files contain one original synthetic member named `payload.txt`. Its
bytes are the ASCII text `Bato nested archive fixture v1` followed by a
single LF. The source modification time was set to 2020-01-01 00:00:00 UTC.

- `payload.rar` was generated with RAR 7.01 using
  `rar a -ma5 -m5 -idq payload.rar payload.txt`.
  SHA-256 `f9efae3098f94b5c5cfc585e9cbdabb6ba17a9de39ea0523fdfd14b00c3df2c6`.
- `payload.7z` was generated with 7-Zip 23.01 using
  `7zz a -t7z -mx=9 -mtm=off -mta=off -mtc=off payload.7z payload.txt`.
  SHA-256 `b60f5d7ed3ee6faee5f5a45da826a95d9922688bb5bbda10a6c344e47085b1d2`.

Each archive was independently extracted at creation and its member bytes
were compared with the source. The Bato tests use the frozen archives as
opaque inputs and verify exact restoration.
