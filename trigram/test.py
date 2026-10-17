from trigram import TrigramModel

tm = TrigramModel(K=100)

print("===== prob =====")
sentences = [
    "但麥科勒姆一邊喝酒一邊吃飯盒一邊討論一邊修改劇本。",
    "吳光翰，字憲甫，江西吉安府安福縣人，中華人民共和國政治人物。",
    "1900年夏季奧林匹克運動會後，慈禧大怒，說要來一場遊戲一場夢。",
    "莊子十分推崇他的脖子和尾巴。"
]
for s in sentences:
    log_p, steps = tm.prob(s)
    print(s)
    # for t in steps:
    #     print(t[0], end=" ")
    # print(steps[-1][1], steps[-1][2]) 
    print(*steps, sep='\n')
    print(f"log P = {log_p:.2f}\n")
