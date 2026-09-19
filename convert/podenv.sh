# Loads pod-level env (incl. the HF_TOKEN RunPod secret) into SSH sessions without storing the value.
export HF_TOKEN="$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^HF_TOKEN=//p")"
export HF_HOME=/workspace/.hf PYTHONUNBUFFERED=1
