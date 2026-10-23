# Failure-case analysis (real test queries)

Run `real-main-01` (level main). Every case is drawn from the test queries in predictions.parquet, ordered by query_id and then picked within each category; no 'nice looking' example was hand-picked. Core models: A0, A1 and B0 / B1 / C0 at the deploy seed 42.

How to read this: **Observation** is a fact read directly from the data; **Hypothesis to test** is only a possible cause, not verified by a controlled experiment — neither feature importance nor case correlation is a causal explanation. Labels are proxy labels (rating ≥ 8), unrated titles are weak negatives, and a 'miss' does not mean the user would dislike the recommendation.

| Category | Occurrences in the test set | Cases shown |
|---|---|---|
| A0 hits, learned models (B1 and C0) miss | 26 | 2 |
| A learned model hits, A0 misses | 518 | 2 |
| All five core models miss | 2399 | 2 |
| Sparse collections (1–4 visible titles) | 600 | 2 |
| Mixed preferences (genre entropy above the validation median) | 1809 | 2 |
| Rule excludes the hidden target (strict mode does not insert it) | 5146 | 3 |
| Strict empty result (candidate count = 0) | 0 | 0 |
| Fewer than 8 candidates | 0 | 0 |

## A0 hits, learned models (B1 and C0) miss

Queries in the test set that fall into this category: 26.
### Case 1: A0 hits, learned models (B1 and C0) miss · query `test-1088591-mal:1840`

- Offline user 1088591 · main group · 77 visible titles · rule R0 (no filter)
- Visible collection: Naruto; School Rumble; Neon Genesis Evangelion; .hack//Sign; DearS; Black Cat (TV); Shuffle!; Air; Fullmetal Alchemist; Vandread; … (77 titles)
- Profile top-10 tags: genre:Comedy, theme:School, genre:Action, genre:Romance, genre:Fantasy, genre:Ecchi, genre:Sci-Fi, genre:Adventure, genre:Drama, genre:Supernatural
- Hidden target: **Zero no Tsukaima: Futatsuki no Kishi** (mal:1840) · public score 7.41 · episodes 12 · popularity rank 415/14182 · tags genre:Action, genre:Adventure, genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:Isekai, theme:Military, theme:School · studio J.C.Staff
- Overlap between the target and the profile top-10 tags: 7 (genre:Action, genre:Adventure, genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:School); the target's studio appears in the visible collection
- Candidates: 14105; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2 | 1 | Urusei Yatsura / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … / Negima!? |
| A1 | 4 | 1 | Urusei Yatsura / Cardcaptor Sakura / Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… |
| B0_seed42 | 9 | 0 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Tenkuu no Shiro Laputa / Cardcaptor Sakura / Zero no Tsukaima: Princ… / Clannad: After Story |
| B1_seed42 | 9 | 0 | One Piece / Zero no Tsukaima: Princ… / Rosario to Vampire Capu2 / Clannad: After Story / Bleach / Prison School / Gintama' / Gintama': Enchousen |
| C0_seed42 | 74 | 0 | Death Note / Ouran Koukou Host Club / Clannad: After Story / Clannad / One Piece / Kaichou wa Maid-sama! / Katekyo Hitman Reborn! / Angel Beats! |

**Observation**: A0's Top-8 mean public score 7.19, mean hits 7.0; the target has public score 7.41 and hits 7. Learned models' rank for the target: B0_seed42 9, B1_seed42 9, C0_seed42 74.

**Hypothesis (to verify)**: the target overlaps the profile tags heavily (hits 7, not below the average of the A0 recommendations), so A0's hits term ranks it high; the learned models lean more on public score (66% of B1's gain is rating_norm) and put titles with a higher score or popularity above it.

### Case 2: A0 hits, learned models (B1 and C0) miss · query `test-1103683-mal:1313`

- Offline user 1103683 · main group · 35 visible titles · rule R0 (no filter)
- Visible collection: Gakuen Alice; Full Moon wo Sagashite; Yumeria; Cardcaptor Sakura; Puchi Pri*Yucie; Psychic Academy; Cardcaptor Sakura Movie 1; Cardcaptor Sakura Movie 2: Fuuin Sareta…; Yu☆Gi☆Oh! Duel Monsters; Yu☆Gi☆Oh! Duel Monsters GX; … (35 titles)
- Profile top-10 tags: genre:Comedy, genre:Fantasy, theme:School, genre:Romance, theme:Mahou Shoujo, genre:Adventure, genre:Drama, genre:Action, theme:Coming of Age, theme:Isekai
- Hidden target: **Digimon Adventure 02** (mal:1313) · public score 7.23 · episodes 50 · popularity rank 1033/14182 · tags genre:Action, genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, theme:Coming of Age, theme:Isekai, theme:Mythology · studio Toei Animation
- Overlap between the target and the profile top-10 tags: 7 (genre:Action, genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, theme:Coming of Age, theme:Isekai); the target's studio appears in the visible collection
- Candidates: 14147; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 8 | 1 | Bishoujo Senshi Sailor … / Cardcaptor Sakura: Clea… / Flip Flappers / Sugar Sugar Rune / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Digimon Adventure 02 |
| A1 | 21 | 0 | Bishoujo Senshi Sailor … / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Cardcaptor Sakura: Clea… / Flip Flappers / Princess Tutu / Fullmetal Alchemist: Br… / Tsubasa: Shunraiki |
| B0_seed42 | 9 | 0 | Fullmetal Alchemist: Br… / Death Note / Fullmetal Alchemist / One Piece / Mononoke Hime / Kaguya-sama wa Kokurase… / Naruto: Shippuuden / Zero no Tsukaima |
| B1_seed42 | 23 | 0 | Ojamajo Doremi Dokkaan! / Ojamajo Doremi Na-i-sho / Ojamajo Doremi / Motto! Ojamajo Doremi / One Piece / Ojamajo Doremi Sharp / Digimon Xros Wars: Aku … / Lovely★Complex |
| C0_seed42 | 133 | 0 | Naruto / Lovely★Complex / Bishoujo Senshi Sailor … / Bleach / One Piece / Junjou Romantica 2 / Fruits Basket / Ouran Koukou Host Club |

**Observation**: A0's Top-8 mean public score 7.49, mean hits 7.0; the target has public score 7.23 and hits 7. Learned models' rank for the target: B0_seed42 9, B1_seed42 23, C0_seed42 133.

**Hypothesis (to verify)**: the target overlaps the profile tags heavily (hits 7, not below the average of the A0 recommendations), so A0's hits term ranks it high; the learned models lean more on public score (66% of B1's gain is rating_norm) and put titles with a higher score or popularity above it.

## A learned model hits, A0 misses

Queries in the test set that fall into this category: 518.
### Case 3: A learned model hits, A0 misses · query `test-101298-mal:1575`

- Offline user 101298 · main group · 6 visible titles · rule R0 (no filter)
- Visible collection: Full Metal Panic!; Full Metal Panic? Fumoffu; Full Metal Panic! The Second Raid; Elfen Lied; Great Teacher Onizuka; Kiba
- Profile top-10 tags: genre:Action, theme:Military, theme:School, genre:Comedy, theme:Delinquents, theme:Mecha, genre:Adventure, genre:Drama, genre:Fantasy, genre:Horror
- Hidden target: **Code Geass: Hangyaku no Lelouch** (mal:1575) · public score 8.70 · episodes 25 · popularity rank 21/14182 · tags genre:Action, genre:Award Winning, genre:Drama, genre:Sci-Fi, theme:Mecha, theme:Military, theme:School, theme:Super Power · studio Sunrise
- Overlap between the target and the profile top-10 tags: 5 (genre:Action, genre:Drama, theme:Mecha, theme:Military, theme:School); the target's studio does not appear in the visible collection
- Candidates: 14176; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 12 | 0 | Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Top wo Nerae! Gunbuster / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Urusei Yatsura / Berserk: Ougon Jidai-he… / Pokemon Movie 01: Mewtw… |
| A1 | 3 | 1 | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Code Geass: Hangyaku no… / Berserk: Ougon Jidai-he… / Code Geass: Hangyaku no… / Top wo Nerae! Gunbuster / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… |
| B0_seed42 | 2 | 1 | Fullmetal Alchemist: Br… / Code Geass: Hangyaku no… / Death Note / Fullmetal Alchemist / Kenpuu Denki Berserk / One Piece / Mononoke Hime / Code Geass: Hangyaku no… |
| B1_seed42 | 7 | 1 | Fullmetal Alchemist: Br… / Death Note / One Piece / Tengen Toppa Gurren Lag… / Violet Evergarden / Suzumiya Haruhi no Shou… / Code Geass: Hangyaku no… / Naruto: Shippuuden |
| C0_seed42 | 5 | 1 | Death Note / Fullmetal Alchemist / Fullmetal Alchemist: Br… / One Piece / Code Geass: Hangyaku no… / Naruto / Code Geass: Hangyaku no… / Naruto: Shippuuden |

**Observation**: A0's Top-8 mean public score 7.93, mean hits 6.0; the target has public score 8.70 and hits 5. Learned models' rank for the target: B0_seed42 2, B1_seed42 7, C0_seed42 5.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.B1 / C0 use signals A0 does not have (studio, episode count, similarity), which may rank it higher; this is a correlational observation, not causal proof.

### Case 4: A learned model hits, A0 misses · query `test-10152-mal:164`

- Offline user 10152 · main group · 22 visible titles · rule R0 (no filter)
- Visible collection: Naruto; .hack//Sign; Aa! Megami-sama! (TV); Ai Yori Aoshi; D.N.Angel; Full Metal Panic!; Mai-HiME; Ayashi no Ceres; Ou Dorobou Jing; Onegai☆Teacher; … (22 titles)
- Profile top-10 tags: genre:Adventure, genre:Fantasy, genre:Action, genre:Comedy, genre:Romance, theme:School, genre:Drama, genre:Supernatural, genre:Award Winning, theme:Coming of Age
- Hidden target: **Mononoke Hime** (mal:164) · public score 8.67 · episodes 1 · popularity rank 105/14182 · tags genre:Action, genre:Adventure, genre:Award Winning, genre:Fantasy · studio Studio Ghibli
- Overlap between the target and the profile top-10 tags: 4 (genre:Action, genre:Adventure, genre:Award Winning, genre:Fantasy); the target's studio appears in the visible collection
- Candidates: 14160; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 136 | 0 | Sennen Joyuu / Cardcaptor Sakura / Nakitai Watashi wa Neko… / Urusei Yatsura Movie 1:… / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Little Busters! Refrain / Fullmetal Alchemist |
| A1 | 26 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Little Busters! Refrain / Fullmetal Alchemist / Fullmetal Alchemist: Br… / Kurenai no Buta |
| B0_seed42 | 3 | 1 | Fullmetal Alchemist: Br… / Fullmetal Alchemist / Mononoke Hime / Toki wo Kakeru Shoujo / Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… |
| B1_seed42 | 1 | 1 | Mononoke Hime / Fullmetal Alchemist: Br… / Kono Subarashii Sekai n… / One Piece / Code Geass: Hangyaku no… / Juuni Kokuki / 86 Part 2 / Akatsuki no Yona |
| C0_seed42 | 32 | 0 | Death Note / One Piece / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Bleach: Sennen Kessen-h… / Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Ouran Koukou Host Club |

**Observation**: A0's Top-8 mean public score 7.94, mean hits 6.5; the target has public score 8.67 and hits 4. Learned models' rank for the target: B0_seed42 3, B1_seed42 1, C0_seed42 32.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.B1 / C0 use signals A0 does not have (studio, episode count, similarity) and the target's studio appears in the visible collection, which may rank it higher; this is a correlational observation, not causal proof.

## All five core models miss

Queries in the test set that fall into this category: 2399.
### Case 5: All five core models miss · query `test-1000595-mal:241`

- Offline user 1000595 · main group · 72 visible titles · rule R0 (no filter)
- Visible collection: Chrno Crusade; Air; Final Approach; Fullmetal Alchemist; Green Green; InuYasha; Happy☆Lesson; Happy☆Lesson (TV); Happy☆Lesson Advance; Bleach; … (72 titles)
- Profile top-10 tags: theme:School, genre:Romance, genre:Comedy, genre:Action, genre:Drama, genre:Ecchi, genre:Sci-Fi, genre:Supernatural, genre:Fantasy, theme:Coming of Age
- Hidden target: **Girls Bravo: First Season** (mal:241) · public score 6.38 · episodes 11 · popularity rank 1457/14182 · tags genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:Isekai, theme:School · studio AIC Spirits
- Overlap between the target and the profile top-10 tags: 5 (genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:School); the target's studio does not appear in the visible collection
- Candidates: 14110; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 292 | 0 | Chobits / Urusei Yatsura Movie 1:… / Blue Seed / Gakusen Toshi Asterisk … / Gakusen Toshi Asterisk / Little Busters! Refrain / Cardcaptor Sakura / Top wo Nerae! Gunbuster |
| A1 | 1513 | 0 | Little Busters! Refrain / Cardcaptor Sakura / Gintama: The Final / Chobits / Kaguya-sama wa Kokurase… / Top wo Nerae! Gunbuster / Bakuman. 2nd Season / Shinsekai yori |
| B0_seed42 | 746 | 0 | Toki wo Kakeru Shoujo / Chobits / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Cardcaptor Sakura / Clannad: After Story / Bakuman. 2nd Season / Mushoku Tensei: Isekai … |
| B1_seed42 | 667 | 0 | Yahari Ore no Seishun L… / High School DxD New / Yahari Ore no Seishun L… / Clannad: After Story / One Piece / Dr. Stone: New World / Prison School / InuYasha: Kanketsu-hen |
| C0_seed42 | 257 | 0 | Ouran Koukou Host Club / Clannad / Clannad: After Story / Lucky☆Star / Death Note / Durarara!! / Kanon (2006) / K-On! |

**Observation**: A0's Top-8 mean public score 7.44, mean hits 6.6; the target has public score 6.38 and hits 5. Learned models' rank for the target: B0_seed42 746, B1_seed42 667, C0_seed42 257.

**Hypothesis (to verify)**: the target's public score is clearly below the average of the A0 recommendations, so an ordering driven mainly by public score may push it down.the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

### Case 6: All five core models miss · query `test-100433-mal:401`

- Offline user 100433 · main group · 9 visible titles · rule R0 (no filter)
- Visible collection: Trigun; Naruto; School Rumble; Rurouni Kenshin: Meiji Kenkaku Romantan…; Rurouni Kenshin: Meiji Kenkaku Romantan; Rurouni Kenshin: Meiji Kenkaku Romantan…; Kidou Senshi Gundam SEED; Bleach; Naruto: Shippuuden
- Profile top-10 tags: genre:Action, genre:Adventure, genre:Romance, genre:Drama, genre:Fantasy, theme:Historical, theme:Martial Arts, theme:Samurai, genre:Comedy, genre:Sci-Fi
- Hidden target: **Rurouni Kenshin: Meiji Kenkaku Romantan - Seisou-hen** (mal:401) · public score 7.97 · episodes 2 · popularity rank 2160/14182 · tags genre:Drama, genre:Romance, theme:Historical, theme:Samurai · studio Studio Deen
- Overlap between the target and the profile top-10 tags: 4 (genre:Drama, genre:Romance, theme:Historical, theme:Samurai); the target's studio appears in the visible collection
- Candidates: 14173; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 277 | 0 | Gintama: The Final / Saber Marionette J / Katanagatari / Jigokuraku / Sennen Joyuu / Cardcaptor Sakura / Vampire Hunter / Kurenai no Buta |
| A1 | 174 | 0 | Gintama: The Final / Gintama° / Gintama' / Gintama': Enchousen / Gintama. / Gintama / Gintama Movie 2: Kanket… / Gintama.: Shirogane no … |
| B0_seed42 | 227 | 0 | Fullmetal Alchemist: Br… / Jigokuraku / Gintama Movie 2: Kanket… / Gintama.: Shirogane no … / Death Note / The First Slam Dunk / Samurai Champloo / Stranger: Mukou Hadan |
| B1_seed42 | 18 | 0 | Fullmetal Alchemist: Br… / Death Note / One Piece / Bleach: Sennen Kessen-h… / Samurai Champloo / Trigun: Badlands Rumble / Code Geass: Hangyaku no… / Cowboy Bebop |
| C0_seed42 | 230 | 0 | One Piece / Fullmetal Alchemist / Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Death Note / Bleach: Sennen Kessen-h… / Code Geass: Hangyaku no… / Dragon Ball Z |

**Observation**: A0's Top-8 mean public score 7.99, mean hits 6.2; the target has public score 7.97 and hits 4. Learned models' rank for the target: B0_seed42 227, B1_seed42 18, C0_seed42 230.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the target is relatively obscure in the catalogue (popularity rank 2160), the main signals of every model relate to public score / popularity, and obscure targets rarely reach the top 8.

## Sparse collections (1–4 visible titles)

Queries in the test set that fall into this category: 600.
### Case 7: Sparse collections (1–4 visible titles) · query `test-100898-mal:63`

- Offline user 100898 · sparse group · 3 visible titles · rule R0 (no filter)
- Visible collection: Yu☆Gi☆Oh! Duel Monsters; Digimon Adventure; Death Note
- Profile top-10 tags: genre:Action, genre:Adventure, genre:Fantasy, genre:Comedy, genre:Supernatural, genre:Suspense, theme:Coming of Age, theme:Historical, theme:Isekai, theme:Mythology
- Hidden target: **DearS** (mal:63) · public score 6.56 · episodes 12 · popularity rank 1451/14182 · tags genre:Comedy, genre:Ecchi, genre:Romance, genre:Sci-Fi · studio Daume
- Overlap between the target and the profile top-10 tags: 1 (genre:Comedy); the target's studio does not appear in the visible collection
- Candidates: 14179; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 7454 | 0 | Digimon Adventure 02 / Dragon Ball Z / Seirei no Moribito / Akatsuki no Yona / Fate/Grand Order: Zetta… / Drifters / Drifters: Special Editi… / Fushigi Yuugi |
| A1 | 7207 | 0 | Dragon Ball Z / Seirei no Moribito / Mo Dao Zu Shi: Wanjie P… / Akatsuki no Yona / Fate/Grand Order: Zetta… / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… / Drifters |
| B0_seed42 | 6679 | 0 | Fullmetal Alchemist: Br… / Ping Pong the Animation / The First Slam Dunk / Akatsuki no Yona / One Piece / Mononoke Hime / Fate/stay night Movie: … / Naruto: Shippuuden |
| B1_seed42 | 3973 | 0 | One Piece / Dragon Ball Z / Hunter x Hunter (2011) / Dragon Ball / Dragon Ball Kai (2014) / Fullmetal Alchemist: Br… / Saint Seiya / Dragon Ball Kai |
| C0_seed42 | 583 | 0 | Naruto / One Piece / Fullmetal Alchemist / Naruto: Shippuuden / Fullmetal Alchemist: Br… / Bleach / Code Geass: Hangyaku no… / Bleach: Sennen Kessen-h… |

**Observation**: A0's Top-8 mean public score 7.83, mean hits 6.1; the target has public score 6.56 and hits 1. Learned models' rank for the target: B0_seed42 6679, B1_seed42 3973, C0_seed42 583.

**Hypothesis (to verify)**: the target's public score is clearly below the average of the A0 recommendations, so an ordering driven mainly by public score may push it down.the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection holds only 3 titles, each contributing 1/3 of the profile frequency, so the tag signal is noisy.

### Case 8: Sparse collections (1–4 visible titles) · query `test-100963-mal:71`

- Offline user 100963 · sparse group · 3 visible titles · rule R0 (no filter)
- Visible collection: D.Gray-man; Shijou Saikyou no Deshi Kenichi; Macross F
- Profile top-10 tags: genre:Action, genre:Adventure, genre:Award Winning, genre:Comedy, genre:Fantasy, genre:Romance, genre:Sci-Fi, theme:Delinquents, theme:Martial Arts, theme:Mecha
- Hidden target: **Full Metal Panic!** (mal:71) · public score 7.61 · episodes 24 · popularity rank 421/14182 · tags genre:Action, genre:Comedy, genre:Sci-Fi, theme:Mecha, theme:Military, theme:School · studio Gonzo
- Overlap between the target and the profile top-10 tags: 4 (genre:Action, genre:Comedy, genre:Sci-Fi, theme:Mecha); the target's studio does not appear in the visible collection
- Candidates: 14179; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 256 | 0 | Saber Marionette J / Turn A Gundam / Pokemon Movie 01: Mewtw… / Dragon Ball Z Movie 12:… / Kidou Butouden G Gundam / Dragon Ball Z Movie 13:… / Dragon Ball Z Movie 08:… / Dragon Ball: Episode of… |
| A1 | 282 | 0 | Tengen Toppa Gurren Lag… / Code Geass: Hangyaku no… / Saber Marionette J / Tenkuu no Shiro Laputa / Sennen Joyuu / Turn A Gundam / Code Geass: Hangyaku no… / Mononoke Hime |
| B0_seed42 | 191 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Death Note / Natsume Yuujinchou Roku / Tenkuu no Shiro Laputa / Code Geass: Hangyaku no… / One Piece / Tengen Toppa Gurren Lag… |
| B1_seed42 | 534 | 0 | Death Note / One Piece / Fullmetal Alchemist: Br… / Macross F Movie 2: Sayo… / Fairy Tail / Bleach / Dragon Ball Z / Fullmetal Alchemist |
| C0_seed42 | 69 | 0 | Naruto / One Piece / Fullmetal Alchemist / Naruto: Shippuuden / Fullmetal Alchemist: Br… / Death Note / Bleach / Code Geass: Hangyaku no… |

**Observation**: A0's Top-8 mean public score 7.47, mean hits 6.1; the target has public score 7.61 and hits 4. Learned models' rank for the target: B0_seed42 191, B1_seed42 534, C0_seed42 69.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection holds only 3 titles, each contributing 1/3 of the profile frequency, so the tag signal is noisy.

## Mixed preferences (genre entropy above the validation median)

Queries in the test set that fall into this category: 1809.
### Case 9: Mixed preferences (genre entropy above the validation median) · query `test-10085-mal:437`

- Offline user 10085 · main group · 36 visible titles · rule R0 (no filter)
- Visible collection: Neon Genesis Evangelion; Neon Genesis Evangelion: The End of Eva…; Koukaku Kidoutai; Akira; Azumanga Daiou The Animation; Blood+; Mononoke Hime; Sen to Chihiro no Kamikakushi; Lodoss-tou Senki; Genshiken; … (36 titles)
- Profile top-10 tags: genre:Award Winning, genre:Drama, genre:Adventure, genre:Sci-Fi, genre:Action, genre:Comedy, genre:Romance, genre:Fantasy, theme:School, theme:Coming of Age
- Hidden target: **Perfect Blue** (mal:437) · public score 8.54 · episodes 1 · popularity rank 288/14182 · tags genre:Avant Garde, genre:Drama, genre:Horror, theme:Adult Cast, theme:Psychological, theme:Showbiz · studio Madhouse
- Overlap between the target and the profile top-10 tags: 1 (genre:Drama); the target's studio appears in the visible collection
- Candidates: 14146; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 4856 | 0 | Urusei Yatsura Movie 1:… / Cardcaptor Sakura / Top wo Nerae! Gunbuster / Turn A Gundam / Urusei Yatsura / Blue Seed / Cardcaptor Sakura Movie… / Urusei Yatsura: Haru da… |
| A1 | 1365 | 0 | Cardcaptor Sakura / Top wo Nerae! Gunbuster / Turn A Gundam / Urusei Yatsura / Cardcaptor Sakura Movie… / Code Geass: Hangyaku no… / Fullmetal Alchemist / Toki wo Kakeru Shoujo |
| B0_seed42 | 638 | 0 | Fullmetal Alchemist: Br… / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Code Geass: Hangyaku no… / Death Note / Evangelion: 1.0 You Are… / One Piece / Code Geass: Hangyaku no… |
| B1_seed42 | 30 | 0 | Toki wo Kakeru Shoujo / Fullmetal Alchemist: Br… / Kaze Tachinu / Tengen Toppa Gurren Lag… / Cardcaptor Sakura / Koukaku Kidoutai: Stand… / Kimi ni Todoke 2nd Seas… / Kimi ni Todoke |
| C0_seed42 | 82 | 0 | Code Geass: Hangyaku no… / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / Fullmetal Alchemist: Br… / Naruto: Shippuuden / Cowboy Bebop / One Piece |

**Observation**: A0's Top-8 mean public score 7.54, mean hits 7.0; the target has public score 8.54 and hits 1. Learned models' rank for the target: B0_seed42 638, B1_seed42 30, C0_seed42 82.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection's genre entropy is above the pre-determined threshold 3.33, the preferences are dispersed, and the top-10 tags hardly represent a single taste.

### Case 10: Mixed preferences (genre entropy above the validation median) · query `test-101103-mal:4081`

- Offline user 101103 · main group · 41 visible titles · rule R0 (no filter)
- Visible collection: Cowboy Bebop; Cowboy Bebop: Tengoku no Tobira; Beck; Shin Kidou Senki Gundam Wing; Fruits Basket; Tenkuu no Escaflowne; Sen to Chihiro no Kamikakushi; Wolf's Rain; Samurai Champloo; FLCL; … (41 titles)
- Profile top-10 tags: genre:Action, genre:Drama, genre:Adventure, genre:Comedy, genre:Mystery, genre:Supernatural, genre:Fantasy, genre:Sci-Fi, genre:Award Winning, theme:Psychological
- Hidden target: **Natsume Yuujinchou** (mal:4081) · public score 8.31 · episodes 13 · popularity rank 393/14182 · tags genre:Drama, genre:Slice of Life, genre:Supernatural, theme:Coming of Age, theme:Iyashikei, theme:School · studio Brain's Base
- Overlap between the target and the profile top-10 tags: 2 (genre:Drama, genre:Supernatural); the target's studio appears in the visible collection
- Candidates: 14141; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2165 | 0 | Serial Experiments Lain / One Piece Film: Red / Pokemon Movie 01: Mewtw… / RahXephon / Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Evangelion: 3.0+1.0 Thr… |
| A1 | 629 | 0 | Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Evangelion: 3.0+1.0 Thr… / Fullmetal Alchemist: Br… / Serial Experiments Lain / Mo Dao Zu Shi: Wanjie P… / Kenpuu Denki Berserk |
| B0_seed42 | 401 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Made in Abyss Movie 3: … / Serial Experiments Lain |
| B1_seed42 | 167 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Tengen Toppa Gurren Lag… / Kidou Senshi Gundam 00 / Code Geass: Hangyaku no… / Danshi Koukousei no Nic… / Danshi Koukousei no Nic… / One Piece |
| C0_seed42 | 307 | 0 | Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Fullmetal Alchemist / Death Note / Fullmetal Alchemist: Br… / Naruto: Shippuuden / One Piece / Tengen Toppa Gurren Lag… |

**Observation**: A0's Top-8 mean public score 8.19, mean hits 5.5; the target has public score 8.31 and hits 2. Learned models' rank for the target: B0_seed42 401, B1_seed42 167, C0_seed42 307.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection's genre entropy is above the pre-determined threshold 3.33, the preferences are dispersed, and the top-10 tags hardly represent a single taste.

## Rule excludes the hidden target (strict mode does not insert it)

Queries in the test set that fall into this category: 5146.
### Case 11: Rule excludes the hidden target (strict mode does not insert it) · query `test-101230-mal:196`

- Offline user 101230 · main group · 105 visible titles · rule R2 (public score ≥ 7.0 and episodes ≤ 26)
- Visible collection: Cowboy Bebop; Aa! Megami-sama! (TV); Chobits; Chrno Crusade; D.N.Angel; Black Cat (TV); Full Metal Panic? Fumoffu; Full Metal Panic! The Second Raid; Shuffle!; Mai-HiME; … (105 titles)
- Profile top-10 tags: genre:Comedy, genre:Romance, theme:School, genre:Action, genre:Fantasy, genre:Ecchi, genre:Supernatural, genre:Drama, genre:Adventure, genre:Sci-Fi
- Hidden target: **Onegai☆Twins** (mal:196) · public score 6.82 · episodes 12 · popularity rank 2089/14182 · tags genre:Comedy, genre:Drama, genre:Sci-Fi, theme:Coming of Age, theme:School · studio Daume
- Overlap between the target and the profile top-10 tags: 4 (genre:Comedy, genre:Drama, genre:Sci-Fi, theme:School); the target's studio appears in the visible collection
- Candidates: 3598; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … / Urusei Yatsura 2: Beaut… |
| A1 | n/a | — | Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Zero no Tsukaima |
| B0_seed42 | n/a | — | Code Geass: Hangyaku no… / Mononoke Hime / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Tenkuu no Shiro Laputa / Zero no Tsukaima / Zero no Tsukaima: Princ… / Clannad: After Story |
| B1_seed42 | n/a | — | Code Geass: Hangyaku no… / Kono Subarashii Sekai n… / Clannad / Dungeon ni Deai wo Moto… / Higurashi no Naku Koro … / Bakuman. 2nd Season / Yuragi-sou no Yuuna-san / Zero no Tsukaima |
| C0_seed42 | n/a | — | Ouran Koukou Host Club / Clannad: After Story / Fruits Basket / Clannad / Junjou Romantica 2 / Lucky☆Star / Vampire Knight Guilty / Lovely★Complex |

**Observation**: A0's Top-8 mean public score 7.23, mean hits 6.9; the target has public score 6.82 and hits 4. Why the rule excludes the target: public score below 7.0.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the target is relatively obscure in the catalogue (popularity rank 2089), the main signals of every model relate to public score / popularity, and obscure targets rarely reach the top 8.

### Case 12: Rule excludes the hidden target (strict mode does not insert it) · query `test-101148-mal:182`

- Offline user 101148 · main group · 12 visible titles · rule R1 (keyword = most common genre in the visible collection) · keyword `action``
- Visible collection: Cowboy Bebop; Cowboy Bebop: Tengoku no Tobira; Trigun; Sen to Chihiro no Kamikakushi; Hellsing; Fate/stay night; Vampire Hunter D (2000); Kyuuketsuhime Miyu (TV); Vampire Hunter D; Higurashi no Naku Koro ni; … (12 titles)
- Profile top-10 tags: genre:Action, genre:Supernatural, theme:Gore, genre:Horror, genre:Sci-Fi, theme:Vampire, theme:Mythology, genre:Adventure, genre:Drama, genre:Fantasy
- Hidden target: **Tenkuu no Escaflowne** (mal:182) · public score 7.67 · episodes 26 · popularity rank 1415/14182 · tags genre:Adventure, genre:Fantasy, genre:Romance, theme:Gore, theme:Isekai, theme:Mecha, theme:Mythology · studio Sunrise
- Overlap between the target and the profile top-10 tags: 4 (genre:Adventure, genre:Fantasy, theme:Gore, theme:Mythology); the target's studio appears in the visible collection
- Candidates: 3909; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Akira / X/1999 / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Blood+ |
| A1 | n/a | — | Akira / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Fullmetal Alchemist: Br… / Mo Dao Zu Shi: Wanjie P… / Kimetsu no Yaiba / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… |
| B0_seed42 | n/a | — | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / One Piece / Mononoke Hime / Fate/stay night Movie: … / Akira / Jigokuraku / Hellsing Ultimate |
| B1_seed42 | n/a | — | Fullmetal Alchemist: Br… / Hellsing Ultimate / Mononoke Hime / Code Geass: Hangyaku no… / Trigun: Badlands Rumble / Rurouni Kenshin: Meiji … / One Piece / Gungrave |
| C0_seed42 | n/a | — | Code Geass: Hangyaku no… / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Piece / Fullmetal Alchemist: Br… / Naruto: Shippuuden / Naruto / Tengen Toppa Gurren Lag… |

**Observation**: A0's Top-8 mean public score 7.82, mean hits 6.4; the target has public score 7.67 and hits 4. Why the rule excludes the target: neither the genres nor the title contain the keyword `action`.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

### Case 13: Rule excludes the hidden target (strict mode does not insert it) · query `test-101534-mal:177`

- Offline user 101534 · main group · 18 visible titles · rule R3 (R1 + R2) · keyword `drama``
- Visible collection: Aa! Megami-sama! (TV); Rozen Maiden; Rozen Maiden: Träumend; Fruits Basket; Fullmetal Alchemist; Loveless; Mahou Sensei Negima!; Gravitation; Matantei Loki Ragnarok; Ouran Koukou Host Club; … (18 titles)
- Profile top-10 tags: genre:Drama, genre:Comedy, genre:Supernatural, genre:Romance, genre:Action, theme:School, genre:Adventure, genre:Fantasy, genre:Mystery, theme:Coming of Age
- Hidden target: **Tsubasa Chronicle** (mal:177) · public score 7.52 · episodes 26 · popularity rank 1005/14182 · tags genre:Action, genre:Adventure, genre:Fantasy, genre:Romance, theme:Isekai, theme:Survival · studio Bee Train
- Overlap between the target and the profile top-10 tags: 4 (genre:Action, genre:Adventure, genre:Fantasy, genre:Romance); the target's studio appears in the visible collection
- Candidates: 886; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Urusei Yatsura Movie 1:… / Blue Seed / Sennen Joyuu / Little Busters! Refrain / Little Busters! EX / Kami no Tou / Sekai Saikou no Ansatsu… / Nakitai Watashi wa Neko… |
| A1 | n/a | — | Sennen Joyuu / Little Busters! Refrain / Mo Dao Zu Shi: Wanjie P… / Kara no Kyoukai Movie 5… / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… / Houseki no Kuni / Wu Liuqi: Xuanwu Guo Pi… |
| B0_seed42 | n/a | — | Kaguya-sama wa Kokurase… / Kara no Kyoukai Movie 5… / Clannad: After Story / Bakuman. 2nd Season / Code Geass: Hangyaku no… / Made in Abyss Movie 3: … / Yuuki Yuuna wa Yuusha d… / Toki wo Kakeru Shoujo |
| B1_seed42 | n/a | — | Clannad: After Story / Little Busters! Refrain / Beck / Dennou Coil / Bakuman. 2nd Season / Toradora! / Chobits / Chihayafuru |
| C0_seed42 | n/a | — | Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Elfen Lied / Clannad: After Story / Vampire Knight Guilty / Junjou Romantica 2 / Chobits / Howl no Ugoku Shiro |

**Observation**: A0's Top-8 mean public score 7.55, mean hits 6.2; the target has public score 7.52 and hits 4. Why the rule excludes the target: neither the genres nor the title contain the keyword `drama`.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

## Strict empty result (candidate count = 0)

Queries in the test set that fall into this category: 0.
This category **did not occur** in this run's test set, so it is not shown (never invented).

## Fewer than 8 candidates

Queries in the test set that fall into this category: 0.
This category **did not occur** in this run's test set, so it is not shown (never invented).

