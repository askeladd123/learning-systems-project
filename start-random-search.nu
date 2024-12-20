
def main [-n:int = 20, --dataset:path = 'data/5x5_small.csv', --epoch-override: int, --tag: string] {  
  pueue parallel --group tsetlin 1
  for i in ..<$n {

    let random_params = poetry run python gen-params-random.py
    let params = if ($epoch_override | is-not-empty) {
      $random_params | from toml | update epochs $epoch_override | to toml
    } else { $random_params }

    let tagstring = if ($tag | is-not-empty) { $" --tag ($tag)" } else { '' }
    pueue add --group tsetlin $"echo '($params)' | poetry run python main.py --data ($dataset)($tagstring)"
  }
}
