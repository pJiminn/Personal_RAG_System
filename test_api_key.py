import os
from openai import OpenAI


def test_openai_api_key():
    """
    OPENAI_API_KEY가 환경변수에 제대로 설정되었는지 테스트하는 스크립트입니다.

    실행 방법 (프로젝트 루트 / 가상환경에서):
      python test_api_key.py
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[❌] OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        print("     .env 파일 또는 시스템 환경변수를 확인하세요.")
        return

    print("[✅] OPENAI_API_KEY 환경변수는 감지되었습니다.")

    client = OpenAI(api_key=api_key)

    try:
        # 가장 간단한 테스트: 짧은 프롬프트로 호출
        print("[…] OpenAI Chat Completion API 호출 테스트 중...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "안녕, 한 줄로 짧게 대답해줘."},
            ],
            max_tokens=20,
        )

        content = response.choices[0].message.content
        print("[✅] API 호출 성공! 응답 내용:")
        print("-----")
        print(content)
        print("-----")
    except Exception as e:
        print("[❌] OpenAI API 호출 중 에러가 발생했습니다.")
        print(f"에러 타입: {type(e).__name__}")
        print(f"에러 메시지: {e}")


if __name__ == "__main__":
    test_openai_api_key()


