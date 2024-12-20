ls runs | get name | each {|dir| open ([$dir, results.toml] | path join) | insert dir $dir } |
  insert n {|row| open ([$row.dir, dataset.toml] | path join) | get dimensions } | 
  move dir n accuracy f1_score precision recall --before training_time_s |
  sort-by f1_score
