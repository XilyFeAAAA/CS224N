from sklearn.model_selection import train_test_split
from pathlib import Path


SEQ_LENGTH = 3
OUTPUT_PATH = Path(__file__).parent
OUTPUT_DEV_PATH = OUTPUT_PATH / "dev.txt"
OUTPUT_TEST_PATH = OUTPUT_PATH / "test.txt"

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir()


if __name__ == "__main__":
    samples = []
    letters = [chr(i) for i in range(97, 123)]
    for i in range(len(letters) - SEQ_LENGTH):
        samples.append(letters[i : i + SEQ_LENGTH + 1])
    
    dev_samples, test_samples = train_test_split(samples, test_size=0.1)
    
    with open(OUTPUT_DEV_PATH, "w") as f:
        for dev in dev_samples:
            f.write(" ".join(dev) + "\n")

    with open(OUTPUT_TEST_PATH, "w") as f:
        for dev in test_samples:
            f.write(" ".join(dev) + "\n")
            
    print("="*80)
    print("数据生成成功")
    print("="*80)
