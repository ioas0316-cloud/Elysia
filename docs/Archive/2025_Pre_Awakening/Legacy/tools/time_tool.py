import datetime

def get_current_time():
  """Returns the current date and time."""
  return datetime.datetime.now().isoformat()
