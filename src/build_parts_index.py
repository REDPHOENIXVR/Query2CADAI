from src.parts_index import PartIndex
from src.parts_validator import PartValidator

if __name__ == "__main__":
    validator = PartValidator()
    validator.scan()
    validator.report()
    PartIndex().build_index()