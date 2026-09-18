# -*- coding: utf-8 -*-
"""
数据库层：SQLite 建表 + 演示种子数据 + 真实数据集加载接口。

真实 MyAnimeList 数据集（Kaggle dbdmobile/myanimelist-dataset）接入方式：
    from database import load_mal_dataset
    load_mal_dataset(
        anime_csv="anime-dataset-2023.csv",
        score_csv="users-score-2023.csv",
        details_csv="users-details-2023.csv",   # 可选
    )
该函数预留为接口，字段映射见函数内注释。
"""
import os
import random
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "anime.db")

# ---------------------------------------------------------------------------
# 种子数据：24 部知名动漫（演示用）。
# 字段设计对齐 MAL 作品表，后续接入真实数据时可无缝替换。
# ---------------------------------------------------------------------------
ANIME_SEED = [
    dict(title="Death Note", title_cn="死亡笔记", an_type="TV", episodes=37, aired="2006",
         score=8.62, popularity=3200000, studios="Madhouse", source="Manga", rating_age="R-17+",
         genres=["悬疑", "心理", "超自然"], themes=["心理战", "智斗", "正义与邪恶"],
         demographics="少年向", franchise="Death Note",
         characters=["夜神月", "L", "琉克", "弥海砂"],
         synopsis="高中生夜神月捡到死神琉克掉落的死亡笔记，写下名字的人就会死去。他企图用笔记肃清世界的罪犯，建立新世界的秩序，而天才侦探L则誓要揭开他的真面目，一场天才之间的心理博弈就此展开。"),
    dict(title="L: Change the World", title_cn="L：最后的23天", an_type="Movie", episodes=1, aired="2008",
         score=7.2, popularity=180000, studios="Madhouse", source="Manga", rating_age="PG-13",
         genres=["悬疑", "心理", "剧情"], themes=["心理战", "智斗", "正义与邪恶"],
         demographics="少年向", franchise="Death Note",
         characters=["L", "夜神月"],
         synopsis="《死亡笔记》的外传剧场版，讲述L在自己生命最后的23天里，与企图毁灭世界的病毒恐怖集团展开最终对决的故事。"),
    dict(title="Shingeki no Kyojin", title_cn="进击的巨人", an_type="TV", episodes=25, aired="2013",
         score=8.56, popularity=3900000, studios="WIT Studio", source="Manga", rating_age="R-17+",
         genres=["动作", "剧情", "奇幻"], themes=["末世", "角色成长", "复仇", "反乌托邦"],
         demographics="少年向", franchise="Attack on Titan",
         characters=["艾伦·耶格尔", "三笠·阿克曼", "阿尔敏", "利威尔"],
         synopsis="人类被逼入巨大的城墙之内苟延残喘，墙外的巨人捕食人类。少年艾伦目睹母亲被巨人吞噬后加入调查兵团，发誓驱逐所有巨人。在血与火的战斗中，世界的真相被一层层揭开。"),
    dict(title="Fullmetal Alchemist: Brotherhood", title_cn="钢之炼金术师 FULLMETAL ALCHEMIST", an_type="TV", episodes=64, aired="2009",
         score=9.09, popularity=3400000, studios="Bones", source="Manga", rating_age="R-17+",
         genres=["动作", "冒险", "剧情", "奇幻"], themes=["角色成长", "兄弟情", "等价交换"],
         demographics="少年向", franchise="Fullmetal Alchemist",
         characters=["爱德华·艾尔利克", "阿尔冯斯·艾尔利克", "罗伊·马斯坦"],
         synopsis="为了复活亡母，兄弟俩触犯炼金术禁忌，爱德华失去左臂右腿，弟弟阿尔失去全身。为找回原来的身体，两人踏上寻找贤者之石的旅程，却逐渐揭开国家背后巨大的阴谋。"),
    dict(title="Steins;Gate", title_cn="命运石之门", an_type="TV", episodes=24, aired="2011",
         score=9.07, popularity=2700000, studios="White Fox", source="Visual novel", rating_age="PG-13",
         genres=["科幻", "悬疑", "剧情"], themes=["时间循环", "心理战", "孤独与牺牲"],
         demographics="少年向", franchise="Steins;Gate",
         characters=["冈部伦太郎", "牧濑红莉栖", "椎名真由理"],
         synopsis="自称疯狂科学家的冈部伦太郎意外发明了能向过去发送短信的机器。一次次改变过去拯救同伴的同时，他也被困在无尽的时间线里，必须做出残酷的抉择。"),
    dict(title="Code Geass: Hangyaku no Lelouch", title_cn="反叛的鲁路修", an_type="TV", episodes=25, aired="2006",
         score=8.7, popularity=2200000, studios="Sunrise", source="Original", rating_age="R-17+",
         genres=["动作", "科幻", "剧情"], themes=["智斗", "复仇", "正义与邪恶", "角色成长"],
         demographics="少年向", franchise="Code Geass",
         characters=["鲁路修·兰佩路基", "朱雀", "C.C."],
         synopsis="不列颠帝国统治下的日本（11区），获得『绝对遵守』之力Geass的皇子鲁路修，戴上假面化名Zero，率领黑色骑士团向帝国复仇，掀起席卷世界的变革。"),
    dict(title="Monster", title_cn="怪物", an_type="TV", episodes=74, aired="2004",
         score=8.88, popularity=900000, studios="Madhouse", source="Manga", rating_age="R-17+",
         genres=["悬疑", "心理", "剧情"], themes=["心理战", "正义与邪恶", "人性"],
         demographics="青年向", franchise="Monster",
         characters=["天马贤三", "约翰", "伦克"],
         synopsis="天才脑外科医生天马救活了一个濒死的男孩，多年后男孩约翰成长为冷血的连环杀人魔。天马被卷入追查约翰的旅途，直面人性深处的黑暗。"),
    dict(title="Hunter x Hunter (2011)", title_cn="全职猎人（2011版）", an_type="TV", episodes=148, aired="2011",
         score=9.03, popularity=2500000, studios="Madhouse", source="Manga", rating_age="PG-13",
         genres=["冒险", "动作", "奇幻"], themes=["角色成长", "友情", "智斗"],
         demographics="少年向", franchise="Hunter x Hunter",
         characters=["小杰", "奇犽", "酷拉皮卡", "雷欧力"],
         synopsis="为了寻找传说中的父亲，少年小杰参加猎人考试，结识了奇犽、酷拉皮卡等伙伴。在贪婪之岛、蚂蚁篇等冒险中，友情与黑暗不断考验着他们。"),
    dict(title="Kimetsu no Yaiba", title_cn="鬼灭之刃", an_type="TV", episodes=26, aired="2019",
         score=8.44, popularity=3100000, studios="ufotable", source="Manga", rating_age="R-17+",
         genres=["动作", "奇幻", "冒险"], themes=["复仇", "兄妹情", "角色成长"],
         demographics="少年向", franchise="Kimetsu no Yaiba",
         characters=["灶门炭治郎", "祢豆子", "我妻善逸", "富冈义勇"],
         synopsis="家人惨遭恶鬼杀害，妹妹祢豆子也变成了鬼。炭治郎为了让妹妹变回人类、斩杀杀害家人的鬼，加入鬼杀队，踏上斩鬼之路。"),
    dict(title="Jujutsu Kaisen", title_cn="咒术回战", an_type="TV", episodes=24, aired="2020",
         score=8.55, popularity=2800000, studios="MAPPA", source="Manga", rating_age="R-17+",
         genres=["动作", "超自然", "奇幻"], themes=["战斗", "角色成长", "诅咒"],
         demographics="少年向", franchise="Jujutsu Kaisen",
         characters=["虎杖悠仁", "五条悟", "伏黑惠", "钉崎野蔷薇"],
         synopsis="吞下了特级咒物『两面宿傩的手指』的虎杖悠仁，与最强咒术师五条悟相遇，进入咒术高专，一边与咒灵战斗一边面对宿傩苏醒的危机。"),
    dict(title="Vinland Saga", title_cn="海盗战记", an_type="TV", episodes=24, aired="2019",
         score=8.75, popularity=1100000, studios="WIT Studio", source="Manga", rating_age="R-17+",
         genres=["动作", "冒险", "剧情", "历史"], themes=["复仇", "角色成长", "反战"],
         demographics="青年向", franchise="Vinland Saga",
         characters=["托尔芬", "阿谢拉德"],
         synopsis="维京少年托尔芬的父亲被佣兵队长阿谢拉德杀害，他为复仇加入对方的队伍。在战火与杀戮中，他逐渐思考暴力与和平的真正意义。"),
    dict(title="Cowboy Bebop", title_cn="星际牛仔", an_type="TV", episodes=26, aired="1998",
         score=8.75, popularity=1200000, studios="Sunrise", source="Original", rating_age="R-17+",
         genres=["科幻", "动作", "冒险", "剧情"], themes=["孤独", "赏金猎人", "爵士"],
         demographics="青年向", franchise="Cowboy Bebop",
         characters=["斯派克", "杰特", "菲", "艾德"],
         synopsis="2071年，太阳系各行星间游荡着赏金猎人。斯派克与杰特驾驶Bebop号，与菲、艾德等伙伴在追捕罪犯的过程中，各自背负着无法逃避的过去。"),
    dict(title="Neon Genesis Evangelion", title_cn="新世纪福音战士", an_type="TV", episodes=26, aired="1995",
         score=8.35, popularity=2000000, studios="Gainax", source="Original", rating_age="PG-13",
         genres=["科幻", "心理", "剧情"], themes=["末世", "孤独", "人性", "心理战"],
         demographics="青年向", franchise="Evangelion",
         characters=["碇真嗣", "绫波丽", "明日香", "葛城美里"],
         synopsis="14岁的碇真嗣被父亲召回第三新东京市，驾驶巨型兵器EVA迎战神秘的使徒。在战斗与人心交错的深渊中，少年们被迫面对自己的脆弱与孤独。"),
    dict(title="Mushishi", title_cn="虫师", an_type="TV", episodes=26, aired="2005",
         score=8.69, popularity=700000, studios="Artland", source="Manga", rating_age="PG-13",
         genres=["奇幻", "治愈", "冒险", "剧情"], themes=["单元剧", "人与自然", "旅途"],
         demographics="青年向", franchise="Mushishi",
         characters=["银古"],
         synopsis="研究超自然存在『虫』的虫师银古，游走于山野村落之间，为人们解决与虫有关的奇异现象。一个个故事温柔而哀伤，诉说人与自然的共存。"),
    dict(title="Natsume Yuujinchou", title_cn="夏目友人帐", an_type="TV", episodes=13, aired="2008",
         score=8.36, popularity=800000, studios="Brain's Base", source="Manga", rating_age="PG-13",
         genres=["超自然", "治愈", "奇幻", "剧情"], themes=["单元剧", "人与妖怪", "温柔"],
         demographics="少女向", franchise="Natsume Yuujinchou",
         characters=["夏目贵志", "猫咪老师"],
         synopsis="能看见妖怪的少年夏目贵志，继承了外婆玲子的『友人帐』，开始把名字归还给妖怪们。在人与妖怪的相遇中，一个个温暖的故事徐徐展开。"),
    dict(title="One Punch Man", title_cn="一拳超人", an_type="TV", episodes=12, aired="2015",
         score=8.49, popularity=3000000, studios="Madhouse", source="Web manga", rating_age="R-17+",
         genres=["动作", "喜剧", "科幻"], themes=["无敌", "英雄", "搞笑"],
         demographics="少年向", franchise="One Punch Man",
         characters=["埼玉", "杰诺斯"],
         synopsis="兴趣使然的英雄埼玉，拥有无论多强的敌人都能一拳解决的力量。变强到无敌之后，他反而感到空虚，在英雄协会的日常中寻找战斗的乐趣。"),
    dict(title="Spy x Family", title_cn="间谍过家家", an_type="TV", episodes=12, aired="2022",
         score=8.44, popularity=2400000, studios="Wit Studio,CloverWorks", source="Manga", rating_age="PG-13",
         genres=["动作", "喜剧", "剧情"], themes=["家庭", "间谍", "搞笑"],
         demographics="少年向", franchise="Spy x Family",
         characters=["劳埃德", "约尔", "阿尼亚"],
         synopsis="顶尖间谍『黄昏』为了完成任务，组建了临时家庭：妻子是杀手，女儿是读心者。互不知晓秘密的三人，上演着温馨又爆笑的家庭日常。"),
    dict(title="Mob Psycho 100", title_cn="灵能百分百", an_type="TV", episodes=12, aired="2016",
         score=8.5, popularity=1300000, studios="Bones", source="Web manga", rating_age="PG-13",
         genres=["动作", "喜剧", "超自然"], themes=["角色成长", "超能力", "青春"],
         demographics="少年向", franchise="Mob Psycho 100",
         characters=["影山茂夫", "灵幻新隆"],
         synopsis="拥有超强超能力却存在感稀薄的中学二年级生『龙套』，一边打工（在可疑的灵能顾问事务所），一边寻找自己能力之外的成长道路。"),
    dict(title="Higurashi no Naku Koro ni", title_cn="寒蝉鸣泣之时", an_type="TV", episodes=26, aired="2006",
         score=7.8, popularity=600000, studios="Studio Deen", source="Visual novel", rating_age="R-17+",
         genres=["悬疑", "恐怖", "超自然"], themes=["时间循环", "乡村", "人性"],
         demographics="青年向", franchise="Higurashi",
         characters=["前原圭一", "龙宫礼奈", "园崎魅音"],
         synopsis="昭和58年，雏见泽村。移居而来的少年前原圭一，逐渐发现村庄绵流祭背后的诡异死亡事件。悲剧不断重演，真相却藏在一次次循环之中。"),
    dict(title="Gintama", title_cn="银魂", an_type="TV", episodes=201, aired="2006",
         score=8.94, popularity=1800000, studios="Sunrise", source="Manga", rating_age="PG-13",
         genres=["动作", "喜剧", "科幻", "剧情"], themes=["武士", "搞笑", "日常", "热血"],
         demographics="少年向", franchise="Gintama",
         characters=["坂田银时", "志村新八", "神乐"],
         synopsis="外星人统治的江户，丧失武之魂的武士坂田银时经营着万事屋，与新八、神乐一起接下各种奇怪的委托，在吐槽与热血之间书写最后的武士道。"),
    dict(title="Kimi no Na wa", title_cn="你的名字。", an_type="Movie", episodes=1, aired="2016",
         score=8.82, popularity=2300000, studios="CoMix Wave Films", source="Original", rating_age="PG-13",
         genres=["爱情", "超自然", "剧情"], themes=["时间", "青春", "命运"],
         demographics="少年向", franchise="Makoto Shinkai",
         characters=["立花泷", "宫水三叶"],
         synopsis="东京少年泷与乡村少女三叶毫无交集，却在梦中互换了身体。跨越时空的两人，努力寻找彼此的名字与命运相会的瞬间。"),
    dict(title="Re:Zero kara Hajimeru Isekai Seikatsu", title_cn="Re:从零开始的异世界生活", an_type="TV", episodes=25, aired="2016",
         score=8.21, popularity=2500000, studios="White Fox", source="Light novel", rating_age="R-17+",
         genres=["奇幻", "悬疑", "剧情"], themes=["时间循环", "角色成长", "死亡回归"],
         demographics="少年向", franchise="Re:Zero",
         characters=["菜月昴", "爱蜜莉雅", "雷姆"],
         synopsis="一无所有的普通少年菜月昴被召唤到异世界，只拥有『死亡回归』的能力。每一次死亡都会回到存档点，他在绝望的轮回中拼死守护重要之人。"),
    dict(title="Shigatsu wa Kimi no Uso", title_cn="四月是你的谎言", an_type="TV", episodes=22, aired="2014",
         score=8.64, popularity=1900000, studios="A-1 Pictures", source="Manga", rating_age="PG-13",
         genres=["爱情", "音乐", "剧情"], themes=["音乐", "青春", "治愈", "离别"],
         demographics="少年向", franchise="Shigatsu",
         characters=["有马公生", "宫园薰"],
         synopsis="因母亲去世而再也听不到琴声的钢琴少年有马公生，遇见了自由奔放的小提琴少女宫园薰。她把他重新带回音乐的世界，也留下了一个隐瞒至今的秘密。"),
    dict(title="Hataraku Saibou", title_cn="工作细胞", an_type="TV", episodes=13, aired="2018",
         score=7.85, popularity=1500000, studios="David Production", source="Manga", rating_age="PG-13",
         genres=["喜剧", "科幻", "教育"], themes=["拟人化", "科普", "日常"],
         demographics="少年向", franchise="Cells at Work",
         characters=["红细胞", "白细胞", "血小板"],
         synopsis="在你的身体里，约37兆个细胞每天都在辛勤工作。红细胞运送氧气，白细胞消灭细菌，血小板修复伤口——以拟人化的方式展现人体内的日常战斗。"),
]

# ---------------------------------------------------------------------------
# 名场面知识库（自建，对应 Scene-level Knowledge Base）。
# 真实场景库需要团队自行扩充，这里先放 8 条演示级名场面。
# ---------------------------------------------------------------------------
SCENE_SEED = [
    dict(anime_title="Death Note", episode=2, keywords="夜神月,L,对峙,谈判,芯片,监视,网球,智斗",
         description="L与Light的第一次正面交锋：大学里网球比赛边坐下的两人，一边试探一边互相怀疑，隔着阳光和一句『我想和你做朋友』暗流涌动。"),
    dict(anime_title="Death Note", episode=25, keywords="L,雨,死亡,脚,红苹果,名场面",
         description="雨夜天台中，L扶着Light的脚背说『你果然是我的第一个朋友』，随后倒下——全剧最著名的离别场景。"),
    dict(anime_title="Shingeki no Kyojin", episode=1, keywords="超大型巨人,城墙,屋顶,母亲,吃掉",
         description="第一集：Wall Maria被突破，超大型巨人从城墙上方俯视，艾伦在屋顶眼看巨人抓起自己的母亲。"),
    dict(anime_title="Shingeki no Kyojin", episode=22, keywords="利威尔,女巨人,树林,战斗,丝带",
         description="女巨人追击战：调查兵团在巨树林中轮流接应，利威尔砍杀女巨人的高速战斗场面。"),
    dict(anime_title="Steins;Gate", episode=12, keywords="真由理,死亡,时间循环,铃羽,选择",
         description="冈部一次次回到过去，却始终无法阻止真由理死亡——时间循环绝望感的顶点。"),
    dict(anime_title="Code Geass", episode=1, keywords="Geass,契约,C.C.,绝对遵守,眼睛",
         description="C.C.与鲁路修缔结契约，Geass之力在眼中觉醒：『Lelouch vi Britannia命令你——去死』。"),
    dict(anime_title="Kimi no Na wa", episode=1, keywords="黄昏之时,陨石,口嚼酒,名字,楼梯",
         description="宫水神社御神体处，三叶与泷在『黄昏之时』跨越时空相见，在掌心写下名字的场面。"),
    dict(anime_title="Fullmetal Alchemist: Brotherhood", episode=4, keywords="人体炼成,真理,门,弟弟,失去",
         description="兄弟俩进行人体炼成，被『真理之门』吞噬：爱德华失去左腿，阿尔失去整个身体。"),
    dict(anime_title="Re:Zero kara Hajimeru Isekai Seikatsu", episode=18, keywords="雷姆,告白,从零开始,黄昏,泪",
         description="雷姆在夕阳下向昴告白：『从这里开始吧——从零开始吧』，被治愈系台词包装的最刀名场面。"),
    dict(anime_title="Code Geass", episode=25, keywords="Zero,镇魂曲,死亡,面具,皇帝,终局",
         description="Zero Requiem：假死的鲁路修以暴君之身死于Zero剑下，用死亡换来世界和平的终局布局。"),
    dict(anime_title="One Punch Man", episode=1, keywords="认真拳,一拳,陨石,秃头披风,必杀",
         description="埼玉对巨大怪人使出『认真系列』之外的普通一拳——无敌与空虚并存的反套路名场面。"),
    dict(anime_title="Shigatsu wa Kimi no Uso", episode=21, keywords="信,演奏,告别,眼泪,最后,宫园薰",
         description="宫园薰最后的演奏与那封迟到的信：『我喜欢你们』——音乐与告别交织的收束场景。"),
]

# 演示用户偏好聚类：用于生成评分种子，让协同过滤有真实信号
_PREFERENCE_CLUSTERS = [
    # (喜欢的 genre 集合, 权重高)
    ({"悬疑", "心理", "剧情"}, ["Death Note", "Monster", "Code Geass: Hangyaku no Lelouch", "Steins;Gate", "Higurashi no Naku Koro ni"]),
    ({"动作", "奇幻", "冒险"}, ["Shingeki no Kyojin", "Hunter x Hunter (2011)", "Kimetsu no Yaiba", "Jujutsu Kaisen", "Vinland Saga", "Fullmetal Alchemist: Brotherhood"]),
    ({"治愈", "超自然", "奇幻"}, ["Mushishi", "Natsume Yuujinchou", "Shigatsu wa Kimi no Uso", "Kimi no Na wa"]),
    ({"科幻", "动作", "喜剧"}, ["Cowboy Bebop", "Neon Genesis Evangelion", "One Punch Man", "Mob Psycho 100", "Spy x Family", "Gintama", "Hataraku Saibou", "Re:Zero kara Hajimeru Isekai Seikatsu"]),
]

USER_SEED = [
    dict(id=1, username="demo", gender="Male", mean_score=8.5),
    dict(id=2, username="visitor", gender="Unknown", mean_score=7.5),
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """建表并写入演示种子数据（已存在则跳过）。"""
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS anime (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            title_cn TEXT NOT NULL DEFAULT '',
            an_type TEXT DEFAULT 'TV',
            episodes INTEGER DEFAULT 0,
            aired TEXT DEFAULT '',
            score REAL DEFAULT 0,
            popularity INTEGER DEFAULT 0,
            studios TEXT DEFAULT '',
            source TEXT DEFAULT '',
            rating_age TEXT DEFAULT '',
            genres TEXT DEFAULT '',        -- 逗号分隔
            themes TEXT DEFAULT '',        -- 逗号分隔
            demographics TEXT DEFAULT '',
            franchise TEXT DEFAULT '',     -- 系列/知识图谱用
            characters TEXT DEFAULT '',    -- 逗号分隔
            synopsis TEXT DEFAULT '',
            mal_id INTEGER DEFAULT 0,      -- MyAnimeList 官方 id（Jikan 补全后回填）
            image_url TEXT DEFAULT ''      -- 封面图（Jikan 补全后回填）
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            gender TEXT DEFAULT 'Unknown',
            mean_score REAL DEFAULT 0,
            password_hash TEXT DEFAULT '',   -- PBKDF2 哈希；空 = 演示用户（无密码）
            created_at TEXT DEFAULT ''       -- 注册时间
        );
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            anime_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            created_at TEXT DEFAULT '',      -- 评分时间（历史记录展示用）
            UNIQUE(user_id, anime_id)
        );
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            anime_id INTEGER NOT NULL,
            folder_id INTEGER DEFAULT 0,     -- 0 = 默认收藏夹
            UNIQUE(user_id, anime_id)
        );
        CREATE TABLE IF NOT EXISTS fav_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anime_title TEXT NOT NULL,
            episode INTEGER DEFAULT 0,
            keywords TEXT DEFAULT '',
            description TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_ratings_anime ON ratings(anime_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
        """
    )

    # 旧库平滑迁移：补充新增列（幂等）
    for tbl, col, typ in [
        ("anime", "mal_id", "INTEGER DEFAULT 0"),
        ("anime", "image_url", "TEXT DEFAULT ''"),
        ("users", "password_hash", "TEXT DEFAULT ''"),
        ("users", "created_at", "TEXT DEFAULT ''"),
        ("ratings", "created_at", "TEXT DEFAULT ''"),
        ("favorites", "folder_id", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # 列已存在

    # 给无时间戳的种子评分回填演示时间（幂等，评分历史页展示用）
    blank = conn.execute("SELECT COUNT(*) FROM ratings WHERE created_at=''").fetchone()[0]
    if blank:
        conn.execute(
            """UPDATE ratings SET created_at = datetime('2026-08-01 12:00:00', '+' || (id * 7) || ' hours')
               WHERE created_at=''""")

    if conn.execute("SELECT COUNT(*) FROM anime").fetchone()[0] == 0:
        for i, a in enumerate(ANIME_SEED, start=1):
            conn.execute(
                """INSERT INTO anime (id,title,title_cn,an_type,episodes,aired,score,popularity,
                   studios,source,rating_age,genres,themes,demographics,franchise,characters,synopsis)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (i, a["title"], a["title_cn"], a["an_type"], a["episodes"], a["aired"], a["score"],
                 a["popularity"], a["studios"], a["source"], a["rating_age"],
                 ",".join(a["genres"]), ",".join(a["themes"]), a["demographics"], a["franchise"],
                 ",".join(a["characters"]), a["synopsis"]),
            )
        print(f"[db] 已写入 {len(ANIME_SEED)} 部演示动漫")

    if conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 0:
        for s in SCENE_SEED:
            conn.execute(
                "INSERT INTO scenes (anime_title,episode,keywords,description) VALUES (?,?,?,?)",
                (s["anime_title"], s["episode"], s["keywords"], s["description"]),
            )

    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        for u in USER_SEED:
            conn.execute(
                "INSERT INTO users (id,username,gender,mean_score) VALUES (?,?,?,?)",
                (u["id"], u["username"], u["gender"], u["mean_score"]),
            )
        _seed_ratings(conn)

    conn.commit()
    conn.close()


def _seed_ratings(conn: sqlite3.Connection, n_users: int = 60):
    """生成 60 个偏好聚类的演示用户评分，让 ICF 有信号可算。"""
    anime_rows = conn.execute("SELECT id, title, genres, score FROM anime").fetchall()
    title_to_id = {r["title"]: r["id"] for r in anime_rows}
    rng = random.Random(42)

    for uid in range(3, 3 + n_users):
        cluster = _PREFERENCE_CLUSTERS[(uid - 3) % len(_PREFERENCE_CLUSTERS)]
        loved_genres, loved_titles = cluster
        rated = set()
        # 高分：聚类核心作品 + 同类作品
        for t in loved_titles:
            aid = title_to_id.get(t)
            if aid and aid not in rated:
                conn.execute("INSERT OR IGNORE INTO ratings (user_id,anime_id,rating) VALUES (?,?,?)",
                             (uid, aid, round(rng.uniform(8.5, 10), 1)))
                rated.add(aid)
        for r in anime_rows:
            if r["id"] in rated:
                continue
            gset = set(r["genres"].split(","))
            if gset & loved_genres and rng.random() < 0.45:
                conn.execute("INSERT OR IGNORE INTO ratings (user_id,anime_id,rating) VALUES (?,?,?)",
                             (uid, r["id"], round(rng.uniform(7, 9.5), 1)))
                rated.add(r["id"])
            elif rng.random() < 0.08:  # 少量低分噪音
                conn.execute("INSERT OR IGNORE INTO ratings (user_id,anime_id,rating) VALUES (?,?,?)",
                             (uid, r["id"], round(rng.uniform(3, 6.5), 1)))
                rated.add(r["id"])

    # demo 用户(1)：收藏+高评分 心理智斗系，演示个性化推荐
    for t, score in [("Death Note", 10), ("Steins;Gate", 9.5), ("Code Geass: Hangyaku no Lelouch", 9.5),
                     ("Monster", 9), ("Shingeki no Kyojin", 9)]:
        aid = title_to_id[t]
        conn.execute("INSERT OR IGNORE INTO ratings (user_id,anime_id,rating) VALUES (1,?,?)", (aid, score))
        conn.execute("INSERT OR IGNORE INTO favorites (user_id,anime_id) VALUES (1,?)", (aid,))

    # visitor 用户(2)：看几部热血战斗
    for t, score in [("Kimetsu no Yaiba", 8.5), ("Jujutsu Kaisen", 8.5), ("One Punch Man", 8)]:
        aid = title_to_id[t]
        conn.execute("INSERT OR IGNORE INTO ratings (user_id,anime_id,rating) VALUES (2,?,?)", (aid, score))
        conn.execute("INSERT OR IGNORE INTO favorites (user_id,anime_id) VALUES (2,?)", (aid,))

    print(f"[db] 已生成 {n_users + 2} 个演示用户及评分")


# ---------------------------------------------------------------------------
# 真实数据集加载接口（预留，接入 Kaggle 数据集时实现）
# ---------------------------------------------------------------------------
def load_mal_dataset(anime_csv: str, score_csv: str, details_csv: str | None = None):
    """接入 Kaggle 'Anime Dataset 2023' 真实数据的接口。

    anime-dataset-2023.csv 列映射建议：
        anime_id -> anime.id, Name -> title, Japanese name/Synonym(中文) -> title_cn,
        Score -> score, Members -> popularity, Genres -> genres, Themes -> themes,
        Studios -> studios, Source -> source, Type -> an_type, Episodes -> episodes,
        Aired -> aired, Rating -> rating_age, Synopsis -> synopsis,
        Producers/Other name 按需丢弃或扩展列。
    users-score-2023.csv 列映射建议：
        Username -> users.username(新增映射表), Anime Title -> anime.title,
        rating -> ratings.rating；watching_status 可扩展 ratings 表增加 status 列。
    注意：
        1) 建议先过滤 Hent*ai 类型与无评分作品（可用作品约 1.2 万部）；
        2) themes/demographics 缺口建议配合 Jikan API 补齐；
        3) 大表请分块读取（pandas chunksize=100000）。
    """
    raise NotImplementedError("接入真实数据集时实现，映射说明见 docstring")


if __name__ == "__main__":
    init_db()
    print("数据库初始化完成:", DB_PATH)
