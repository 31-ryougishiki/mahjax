# MahJax PyTorch 测试计划

## 1. 单元测试：数据层

### 1.1 牌面 (tile.py)
| ID | 用例 | 输入 | 预期 |
|:---|:---|:---|:---|
| T01 | 普通牌型转换 | `to_tile_type(4)` → 4, `to_tile_type(34)` → 4 | 红5m→普通5m |
| T02 | 赤牌检测 | `is_tile_red(34)` → True, `is_tile_red(4)` → False | |
| T03 | 红牌转普通 | `to_tile_type(35)` → 13, `to_tile_type(36)` → 22 | |
| T04 | 幺九牌判断 | `is_yaochu(0)`→True, `is_yaochu(4)`→False, `is_yaochu(27)`→True | |
| T05 | FROM_TILE_ID_TO_TILE | 136个tile_id → tile映射，红5在正确位置 | 3张红5, 133张普通 |
| T06 | 牌河编码 | add_discard → decode_river → tile/riichi/tsumogiri一致 | 位运算正确 |
| T07 | 牌河鸣牌编码 | add_meld(PON) → decode → meld_type=1, gray=True | |

### 1.2 面子 (meld.py)
| ID | 用例 | 输入 | 预期 |
|:---|:---|:---|:---|
| M01 | 碰编码 | `init(PON, target, src)` → decode → action/target/src一致 | |
| M02 | 吃编码 | `init(CHI_L, target, src)` → `_chi_index`=0 | |
| M03 | 杠编码 | `init(OPEN_KAN, target, src)` → `is_kan`=True | |
| M04 | 暗杠/加杠 | `init(37+t, t, 0)` → src=0 → `is_closed_kan`=True | |
| M05 | 空面子 | `EMPTY_MELD` → `is_empty`=True, action=-1, target=-1 | |
| M06 | 赤牌面子 | `is_target_red` 正确检测 | |
| M07 | 符号算 | `fu(meld)` → 碰=2, 明杠=8, 暗杠=16, 幺九×2 | |
| M08 | 含赤检测 | PON_RED→True, CHI_L_RED→True, OPEN_KAN(5m)→True | |

### 1.3 手牌 (hand.py)
| ID | 用例 | 输入 | 预期 |
|:---|:---|:---|:---|
| H01 | 摸牌 | `add(hand, tile)` → hand[tile]+=1 | |
| H02 | 切牌 | `sub(hand, tile)` → hand[tile]-=1 | |
| H03 | 37型→34型 | `to_34(hand_with_red)` → 红5合并到黑5 | |
| H04 | 碰判定(普通) | 手牌已有2张 → `can_no_red_pon`=True | |
| H05 | 碰判定(赤) | 有黑5m×1 + 红5m×1 → `can_red_pon`=True | |
| H06 | 吃判定(左) | 手牌有t+1,t+2 → `can_chi(CHI_L)`=True | |
| H07 | 吃判定(中) | 手牌有t-1,t+1 → `can_chi(CHI_M)`=True | |
| H08 | 吃判定(右) | 手牌有t-2,t-1 → `can_chi(CHI_R)`=True | |
| H09 | 吃判定(赤左) | 有t+1红5 → `can_red_chi(CHI_L_RED)`=True | |
| H10 | 明杠判定 | 手牌3张 → `can_open_kan`=True | |
| H11 | 暗杠判定 | 手牌4张 → `can_closed_kan`=True | |
| H12 | 加杠判定 | 手牌1张+已有碰 → `can_added_kan`=True | |
| H13 | 听牌判定 | shanten=0 → `is_tenpai`=True | |
| H14 | 和牌判定(通常) | 完成形+head → `can_tsumo`=True | |
| H15 | 和牌判定(七对子) | 7对 → `can_tsumo`=True | |
| H16 | 和牌判定(国士) | 13种各1+1对 → `can_tsumo`=True | |
| H17 | 荣和判定 | 听牌+荣和牌 → `can_ron`=True | |
| H18 | 立直判定 | 门清听牌 → `can_riichi`=True | |
| H19 | 九种九牌 | 9种以上幺九 → `can_kyuushu`=True | |
| H20 | 暗杠后听牌不变 | `can_closed_kan_after_riichi` | |
| H21 | 碰执行(普通) | `pon_no_red` → 手牌-2 | |
| H22 | 碰执行(赤) | `pon_red` → 黑5-1, 红5-1 | |
| H23 | 吃执行 | `chi_no_red(CHI_L, t)` → 手牌t+1-1, t+2-1 | |
| H24 | 杠执行 | `open_kan` → 手牌-3, `closed_kan` → -4, `added_kan` → -1 | |

### 1.4 向听 (shanten.py)
| ID | 用例 | 输入 | 预期 |
|:---|:---|:---|:---|
| S01 | 完整手牌 | 4组+雀头 → `number`=-1 | |
| S02 | 一向听 | 3组+雀头+1浮牌 → `number`=0 → 标准=1 | |
| S03 | 两向听 | 2组+雀头+2浮牌 → | |
| S04 | 七对子一向听 | 5对+3散 → `seven_pairs`=2 | |
| S05 | 国士无双一向听 | 12种各1+1对 → `thirteen_orphan`=1 | |
| S06 | 振听不影响向听 | furiten状态 → 向听数相同 | |

---

## 2. 单元测试：役种系统 (yaku.py)

### 2.1 一番役
| ID | 用例 | 手牌 | 预期 |
|:---|:---|:---|:---|
| Y01 | 立直 | 门清听牌+立直棒 | fan≥1, riichi=True |
| Y02 | 一发 | 立直后1巡内和 | ippatsu=True |
| Y03 | 门前清自摸 | 门清+自摸 | fully_concealed=True |
| Y04 | 平和 | 门清+顺子×4+两面听+非役牌雀头 | pinfu=True, fu=20(自摸)/30(荣) |
| Y05 | 断幺九 | 无幺九牌 | all_simples=True |
| Y06 | 一杯口 | 同顺子2组 | pure_double_chis=True |
| Y07 | 役牌(场风) | 场风刻子 | prevalent_wind |
| Y08 | 役牌(自风) | 自风刻子 | seat_wind |
| Y09 | 役牌(三元) | 白/发/中刻子 | dragon=True |
| Y10 | 海底捞月 | 最后1张自摸 | bottom_of_the_sea |
| Y11 | 河底捞鱼 | 最后1张荣和 | bottom_of_the_river |
| Y12 | 岭上开花 | 杠后摸牌和 | after_kan |
| Y13 | 抢杠 | 加杠时荣和 | robbing_kan |
| Y14 | 两立直 | 第一巡立直 | double_riichi=True |

### 2.2 二番役
| ID | 用例 | 预期 |
|:---|:---|:---|
| Y15 | 三色同顺 | 万/筒/索同数字顺子 → fan≥2 |
| Y16 | 三色同刻 | 万/筒/索同数字刻子 → fan=2 |
| Y17 | 对对和 | 刻子×4+雀头 → all_pons=True |
| Y18 | 三暗刻 | 暗刻×3 → three_concealed_pons |
| Y19 | 三杠子 | 杠子×3 → three_kans |
| Y20 | 七对子 | 对子×7 → seven_pairs, fu=25 |
| Y21 | 混全带幺九 | 所有组都含幺九 → outside_hand, fan=2(门清)/1(副露) |
| Y22 | 一气通贯 | 同色123+456+789 → pure_straight |
| Y23 | 全带幺 | 所有组都带幺九+无字牌 → terminals_in_all_sets |

### 2.3 三番以上
| ID | 用例 | 预期 |
|:---|:---|:---|
| Y24 | 二杯口 | 一杯口×2 → twice_pure_double_chis, fan=3 |
| Y25 | 混一色 | 一色+字牌 → half_flush |
| Y26 | 纯全带幺九 | 全带幺九+无字 → fan=3(门清) |
| Y27 | 清一色 | 同色 → full_flush, fan=6(门清)/5(副露) |

### 2.4 満貫/役満
| ID | 用例 | 预期 |
|:---|:---|:---|
| Y28 | 四暗刻 | 暗刻×4 → yakuman |
| Y29 | 国士无双 | 13种各1+任意1对 → yakuman |
| Y30 | 大三元 | 白+发+中刻子 → yakuman |
| Y31 | 四喜和 | 四风刻 → double yakuman |
| Y32 | 小四喜 | 三风刻+风雀头 → yakuman |
| Y33 | 字一色 | 全部字牌 → yakuman |
| Y34 | 清老头 | 全部幺九 → yakuman |
| Y35 | 绿一色 | 全部绿色牌 → yakuman |
| Y36 | 九莲宝灯 | 同色1112345678999+任意 → yakuman |
| Y37 | 四杠子 | 杠子×4 → yakuman |

### 2.5 符数计算
| ID | 用例 | 预期 |
|:---|:---|:---|
| F01 | 平和自摸 | 20符 |
| F02 | 平和荣和 | 30符 |
| F03 | 七对子 | 25符 |
| F04 | 暗刻×1 | +4符(中张)/+8符(幺九) |
| F05 | 明刻×1 | +2符(中张)/+4符(幺九) |
| F06 | 暗杠 | +16符(中张)/+32符(幺九) |
| F07 | 明杠 | +8符(中张)/+16符(幺九) |
| F08 | 役牌雀头 | +2符 |
| F09 | 连风雀头 | +4符 |
| F10 | 边张/坎张/单骑听牌 | +2符 |

---

## 3. 集成测试：环境 (env.py)

### 3.1 游戏流程
| ID | 用例 | 验证点 |
|:---|:---|:---|
| E01 | Init | 4家13张+庄家14张, dora揭开, 庄家先行动 |
| E02 | 正常摸打 | draw→discard→next_player, 牌墙递减 |
| E03 | 碰 | discard→other pon→remove_tiles→pon_player_discard |
| E04 | 吃(左) | discard→chi(CHI_L)→remove_tiles→chi_player_discard |
| E05 | 吃(中/右) | 中吃/右吃同样流程 |
| E06 | 明杠 | discard→open_kan→flip_dora→rinshan_draw |
| E07 | 暗杠 | own_turn→closed_kan→flip_dora→rinshan_draw |
| E08 | 加杠 | own_turn(有碰)→added_kan→flip_dora→rinshan_draw |
| E09 | 荣和 | discard→other_ron→game_end |
| E10 | 自摸 | own_turn→tsumo→game_end |
| E11 | 立直 | own_turn→riichi→discard→can_only_discard |
| E12 | 流局(荒牌平局) | 牌墙耗尽→ryukyoku→tenpai/noten结算 |
| E13 | 九种九牌 | 第一巡→kyuushu→abortive_draw |

### 3.2 边界条件
| ID | 用例 | 验证点 |
|:---|:---|:---|
| E14 | 双响炮 | 两人同时ron→allow_double_ron→两家都得点 |
| E15 | 枪杠 | 加杠时ron→robbing_kan→chankan判 |
| E16 | 振听(舍牌) | 听牌中切过→furiten_by_discard→不能ron |
| E17 | 振听(过水) | 放过ron→furiten_by_pass→暂时不能ron |
| E18 | 一发消 | 立直后鸣牌→ippatsu消失 |
| E19 | 杠dora | 杠→翻dora→最多5张 |
| E20 | 王牌耗尽 | 5杠→no_more_kan |
| E21 | 立直后不能鸣牌 | riichi后→只能切牌 |
| E22 | 立直后暗杠 | 不影响听牌→可以暗杠 |
| E23 | 分数不足立直 | score<1000→不能riichi |
| E24 | 四风连打 | 第一巡4人打同风→special_abortive |
| E25 | 四家立直 | 4人立直→special_abortive |
| E26 | 包牌 | 大明杠→对方暗刻已露→pao判定 |
| E27 | 赤牌含分 | 红5→dora→fan+1 per red |
| E28 | 流局满贯 | 舍牌全是幺九+未被鸣→nagashi_mangan |

### 3.3 得点计算
| ID | 用例 | fan/fu | 预期得点(子/亲) |
|:---|:---|:---|:---|
| P01 | 1翻30符 | 1/30 | 1000/1500(荣), 300/500(自摸子), 500all(自摸亲) |
| P02 | 2翻30符 | 2/30 | 2000/2900 |
| P03 | 3翻30符 | 3/30 | 3900/5800 |
| P04 | 4翻30符 | 4/30 | 7700/11600 |
| P05 | 5翻(满贯) | 5/任意 | 8000/12000 |
| P06 | 平和自摸 | 1/20 | 400/700(子), 700all(亲) 实际上自摸平和不计30符 |
| P07 | 七对子 | 2/25 | 1600/2400 |
| P08 | 跳满 | 6-7翻 | 12000/18000 |
| P09 | 倍满 | 8-10翻 | 16000/24000 |
| P10 | 三倍满 | 11-12翻 | 24000/36000 |
| P11 | 役满 | 役满 | 32000/48000 |
| P12 | 两倍役满 | 2×役满 | 64000/96000 |

### 3.4 多局流程
| ID | 用例 | 验证点 |
|:---|:---|:---|
| R01 | 东风战(east) | 4局, dealer_win→renchan |
| R02 | 半庄战(half) | 8局(东4+南4) |
| R03 | 轮庄 | non-dealer_win→dealer_rotate |
| R04 | 本场 | 庄家连庄→honba+1 |
| R05 | 供託 | riichi→kyotaku+1→和了者获得 |
| R06 | 流局供託 | 流局→kyotaku转入下一局 |
| R07 | 终局顺位点 | uma=[30,10,-10,-30] |
| R08 | auto vs dummy_share | 两种next_round_style等价 |

---

## 4. 训练管道集成测试

| ID | 用例 | 验证点 |
|:---|:---|:---|
| C01 | 数据收集无脏数据 | 所有样本action在mask内 |
| C02 | BC训练收敛 | loss下降, acc上升 |
| C03 | PPO训练不崩溃 | loss不NaN, entropy下降 |
| C04 | 模型保存/加载 | save→load→推理一致 |
| C05 | 相同seed可复现 | 两次run→相同loss |
| C06 | 评估统计合理 | avg_rank在合理范围, hora_rate>0 |
| C07 | 1v3评估 | agent_vs_baseline统计指标 |

---

## 5. 压测 / 鲁棒性

| ID | 用例 | 验证点 |
|:---|:---|:---|
| S01 | 100局不崩溃 | random×4 → 0 crash |
| S02 | 1000局数据采集 | 内存不泄漏,OOM不出现 |
| S03 | 非法动作处理 | 传入非法action→游戏终止+惩罚 |
| S04 | max_steps限制 | 长局不无限循环 |
