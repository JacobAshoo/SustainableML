from codecarbon import EmissionsTracker

def start(name):
  t = EmissionsTracker(
    project_name=name,
    output_dir=".",
    output_file=f"codecarbon_{name}.csv",
    log_level="error"
  )
  t.start()
  return t

def stop(t):
  return t.stop()
