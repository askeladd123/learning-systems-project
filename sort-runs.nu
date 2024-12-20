ls runs | get name | each {|dir| open ([$dir, results.toml] | path join) | insert dir $dir } | 
  move dir accuracy f1_score precision recall --before training_time_s |
  sort-by f1_score
