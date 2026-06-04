import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["YEAR"] = "2024"

from api.pitchers import process_pitching_data





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
