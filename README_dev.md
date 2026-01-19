# 최신 원격 상태 받기
git fetch --all --prune

# 돌아가고 싶은 커밋 해시 확인 (원본이 upstream/main이라면 거기서 찾아도 됨)
git log --oneline --decorate --graph
> origin/kibona-from-cde84fb 는 원격-추적 브랜치

# 여기에 작업하려면 같은 위치를 가리키는 로컬 브랜치를 만들어서 전환해야 해요.
git fetch origin
git switch -c kibona-from-cde84fb --track origin/kibona-from-cde84fb
> Check: git branch -vv

# 수정 & 커밋
git status            # 변경 확인
git add -A            # 전체 스테이징(또는 파일 지정: git add path/to/file)
git commit -m "메시지: 무엇을/왜 바꿨는지"


# 푸시 (업스트림 설정까지)
git push -u origin kibona-from-cde84fb

# 참고: 만약 실수로 git switch --detach origin/kibona-from-cde84fb 로 들어갔다면 커밋이 안 됩니다. 이때는:
> git switch -c kibona-from-cde84fb   # 현재 위치에서 브랜치 만들기



