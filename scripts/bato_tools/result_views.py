"""Read-only SQL projections of canonical, independently identified measurements."""


def install_views(connection):
    fields = ['target', 'input_sha256', 'raw_size', 'comp_size', 'ratio', 'status',
              'verified_lossless', 'error', 'measured_at', 'effective_settings', 'compression_profile',
              'comp_time_s', 'decomp_time_s', 'comp_cpu_s', 'decomp_cpu_s',
              'comp_mb_s', 'decomp_mb_s', 'comp_peak_rss_kib', 'decomp_peak_rss_kib',
              'comp_sampled_tree_peak_rss_kib', 'decomp_sampled_tree_peak_rss_kib',
              'cpu_scope', 'tree_rss_scope']
    projected = ',\n'.join(f"json_extract(c.record, '$.{field}') AS {field}" for field in fields)
    connection.execute(f'''CREATE VIEW IF NOT EXISTS measurement_results AS
        SELECT c.run AS run_id, c.id AS case_id,
               json_extract(c.record, '$.workload_id') AS workload_id,
               json_extract(c.record, '$.track') AS track,
               json_extract(c.record, '$.codec') AS codec,
               {projected},
               json_extract(r.record, '$.environment') AS hardware,
               json_extract(r.record, '$.resource_policy') AS resource_policy,
               json_extract(r.record, '$.requested_threads') AS workers,
               json_extract(r.record, '$.label') AS version_label,
               json_extract(r.record, '$.checkout_revision') AS checkout_revision,
               json_extract(r.record, '$.binary_build_revision') AS binary_build_revision,
               (SELECT json_extract(j.value, '$.binary_sha256') FROM json_each(r.record, '$.codecs') j
                WHERE json_extract(j.value, '$.id')=json_extract(c.record, '$.codec')) AS binary_sha256,
               c.record AS case_record, r.record AS run_record
        FROM cases c JOIN runs r ON r.run=c.run''')
    connection.execute("""CREATE VIEW IF NOT EXISTS bato_results AS
        SELECT * FROM measurement_results WHERE codec LIKE 'bato%'""")
    columns = []
    for codec, name in [('zip', 'zip'), ('gzip', 'gzip'), ('7z', 'seven_zip'), ('rar', 'rar'),
                        ('paq8px', 'paq8px'), ('cmix', 'cmix'), ('nncp', 'nncp')]:
        columns.append(f"""CASE WHEN SUM(codec='{codec}')=1 THEN
            MAX(CASE WHEN codec='{codec}' AND status='PASS & VERIFIED' AND verified_lossless=1
                     THEN comp_size END) END AS {name}_bytes""")
    connection.execute(f'''CREATE VIEW IF NOT EXISTS competitor_results AS
        SELECT run_id, workload_id, track, target AS filename, input_sha256 AS sha256,
               raw_size AS original_bytes, compression_profile, hardware, resource_policy,
               {','.join(columns)},
               json_group_array(json_object('codec',codec,'status',status,'error',error,
                   'verified',verified_lossless,'settings',json(effective_settings))) AS codec_results
        FROM measurement_results WHERE codec NOT LIKE 'bato%'
        GROUP BY run_id,workload_id,track,target,input_sha256,raw_size,compression_profile,hardware,resource_policy''')
