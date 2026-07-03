"""
데이터 생성 → MLP/LSTM 학습 일괄 실행 스크립트
"""

import subprocess
import sys


def run(script: str):
    print(f"\n{'=' * 50}\n실행: {script}\n{'=' * 50}")
    result = subprocess.run([sys.executable, script], cwd=__import__("os").path.dirname(__file__))
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    base = __import__("os").path.dirname(__file__)
    run(__import__("os").path.join(base, "generate_data.py"))
    run(__import__("os").path.join(base, "train_mlp.py"))
    run(__import__("os").path.join(base, "train_lstm.py"))
    print("\n모든 설정이 완료되었습니다. 아래 명령으로 앱을 실행하세요:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
