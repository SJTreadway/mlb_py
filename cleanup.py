import os

DIRS = ['data/bat', 'data/daily', 'data/pitch', 'data/results']

def cleanup_directory():
  for directory in DIRS:
    if os.path.exists(directory):
      for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        except Exception as e:
          print(f'Failed to delete {file_path}. Reason: {e}')
    print(f'✅ {directory} completed 🧹')

if __name__ == "__main__":
  cleanup_directory()