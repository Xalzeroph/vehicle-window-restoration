# 2026.11.08 benchmark 鏁版嵁 QC 涓庡叆搴撴祦绋?v2锛坴1.2锛?
> 瑙掕壊锛氭暟鎹绾匡紙synth-dev锛夈€傚彧鍋氥€岄噰闆嗘暟鎹?鈫?QC 鈫?閿佸畾 鈫?benchmark 闆嗐€嶏紱涓嶈缁冦€佷笉鏀硅鏂囥€佷笉璇勬祴銆佷笉 commit銆?> 涓昏剼鏈細`scripts/qc_bench_1108.py`锛?*v1.2**锛孲CHEMA `qc_bench_1108/v2`锛涙湰鍦伴暅鍍?`code/local_tools/qc_bench_1108.py`锛夛紱
> 杈呭姪鑴氭湰锛歚scripts/qc_cluster_map_1108.py`锛堜袱妗?union-find 绨囪〃锛夈€乣scripts/validity_carwind_1108.py`锛堢湡瀹炶溅绐?validity study intake锛夈€?> v1锛坄2026.11.06_鏂伴噰闆嗘暟鎹甉C涓庨攣瀹氭祦绋?md` + `scripts/qc_intake_1106.py`锛夌户缁敤浜?A/B 缁勯€愬鍏ュ簱銆?> 鏁版嵁鐜扮姸锛?026-11-08锛夛細鏂?A/B 閲囬泦涓?real-car validity 鏁版嵁鍧囨湭鍥炰紶锛涙湰鏂囦欢鏄?*灏辩华浜や粯**锛屽共璺戝彧鍐?`/tmp`锛屾湭鍐欑敓浜х洰褰曘€?
## 0. 涓€鍙ヨ瘽鍙ｅ緞
**鍑犱綍 shift鈮?.0 px 纭孩绾匡紱鍏夊害 affine primary `|g_affine鈭?|鈮?.05`锛堣秴鍒?REJECT锛夛紝median g `|g鈭?|鈮?.15` 浠呰瘖鏂?*锛?resid 鍒嗙骇锛堚墹1.2 姝ｅ父 / 1.2鈥?.8 REVIEW / >1.8 REJECT锛?*1.2 涓嶆斁瀹?*锛夛紱鍙嶅皠澶急 ref_abs_mean<2.0 鈫?REVIEW锛?榛戝竷鍏ョ敾銆侀殣绉佸€欓€夈€佽鍙己澶?鈫?REVIEW锛涘唴瀹归噸澶?娉勬紡鎸?md5+dHash+鐩稿叧鏍告煡锛?*鍙湅鍥涙€侊紝浠庝笉鎸夋ā鍨嬪垎鏁扮瓫鏁版嵁**銆?
## 1. 鎶曢€掔洰褰曚笌 meta.json锛堝繀濉級
```
<batch_root>/meta.json
  A/<scene>/I_*.png + GT.png   # group 甯冨眬锛堟柊閲囬泦鎺ㄨ崘锛欰=瀹ゅ唴銆丅=鎴峰琛楅亾锛夛紱鏀寔 pool锛欼/ T/ gt/ R/ omega/
```
- 姣忓満鏅細I锛堣緭鍏ュ抚锛? GT锛堥粦甯冮伄婊℃涓綅甯э級锛屽悓鑶滃悓鐩告満銆侀攣鏇濆厜/鐧藉钩琛?鐒﹁窛锛汫T 涓嶅敮涓€ 鈫?REVIEW銆?- `meta.json` 蹇呭～锛歚scene, group(A/B), domain_desc, film_id, film_glass_spec(瑙?搂10), camera, location_id, date, gt_frame, g, shift, license`锛?  鍙€?`omega_frac, is_dev4, privacy_note, fps, exposure_lock`銆傝鍙細`research-only | internal | cc-by-nc-4.0` 绛夈€?- 閲囬泦绾㈢嚎锛氶粦甯冧笉寰楀叆 I 鐢婚潰锛涗笌 1019/璁粌姹犱笉鍚屽湴鐐癸紱A 缁勫鍐?6鈥? 鍦烘櫙銆丅 缁勬埛澶?4鈥? 鍦烘櫙锛屾瘡鍦烘櫙 2鈥? 瀵广€?
## 2. 涓€閿祦绋嬶紙鏈嶅姟鍣?CPU锛汫PU 涓€寰嬩笉鐢級
```bash
S=/DataDisk/zmy_code/TintedGlass_Enhance/scripts/qc_bench_1108.py; C=/DataDisk/zmy_code/TintedGlass_Enhance/scripts/qc_cluster_map_1108.py
V=/DataDisk/zmy_code/TintedGlass_Enhance/scripts/validity_carwind_1108.py; ROOT=/DataDisk/zmy_code/TintedGlass_Enhance
PY=<conda-env>/bin/python3; export HOME=<home>; cd <project-root>; O=/tmp/ri_dev/synthdev_1108
$PY $S --mode selftest --tmpdir $O/selftest --out $O/selftest_out                  # 1) 鑷锛坴1.2 55/55锛?$PY $S --mode scan  --root <batch> --layout auto --out $O/scan                       # 2) 甯冨眬/鏂囦欢鏁?鍞竴鍐呭
$PY $S --mode qc    --root <batch> --layout auto --out $O/qc --workers 4 --ref panel1019=$ROOT/data/test_sets/real_pair_1019
# 4) 浜哄伐鐩 REVIEW锛坬c_report.md + privacy_flags/ + cloth_flags/锛夆啋 浜哄伐鍐欑瀛楃櫧鍚嶅崟锛沚uild 榛樿鍙敹 PASS/WARN
$PY $S --mode build --root <batch> --layout auto --dst <new_bench> --out $O/build \
    --candidates-review _release/whitelists/<batch>_review_signoff.json --release-mode internal \
    --compat-against $ROOT/data/test_sets/real_pair_nov02_1108/pairs_manifest.json --overwrite
$PY $C --root <new_bench> --out $O/clusters                                          # 5) 涓ゆ。 cluster锛堝璁＄敤锛?$PY $S --mode lock   --dst <new_bench> --out <lock_dir> --operator-note <璋乢鍝ぉ_渚濇嵁>  # 6) 鍐荤粨
$PY $S --mode verify --dst <new_bench> --out <lock_dir>                              #    閫愭枃浠?sha256锛涚鏀?exit 1
```
- **鐧藉悕鍗曪紙v1.1锛屼汉宸ュ啓銆佺姝㈣剼鏈敓鎴愶級**锛歚_release/whitelists/<batch>_review_signoff.json`锛屽彧澧炰笉鏀癸紙閲嶇=鏂版枃浠讹級锛?  `{operator, date, note, keys[], list_sha256[, owner_sign]}`锛沗list_sha256=sha256(json.dumps(sorted(keys), ensure_ascii=False, separators=(',',':')))`锛岀己澶?涓嶇纭敊璇€?  绛惧瓧鏉冮檺锛歩nternal 鎵规 `operator="lead"`锛沺ublic 鎵规椤绘暟鎹墍鏈夎€?co-sign锛坄owner_sign`锛夛紝缂哄垯 `release_gate.blocked=true`銆?- build锛歊EVIEW 椤诲懡涓櫧鍚嶅崟锛?*REJECT 姘镐笉鏀?*锛沗--strict`=鍙敹 PASS锛沗--release-mode internal|public`锛坧ublic 绂?missing-license銆佺 inline R锛夛紱
  `--r-mode inline|aux|none`锛圧 鏄及璁″眰涓嶆槸 GT锛歱ublic 寮哄埗 aux/ + 鍏嶈矗 README锛屼富鍖呬笉鍚?R锛夈€傜敓浜?`dst` 鍙湪 Lead 鎵瑰噯鍚庡啓锛屽共璺戜竴寰?`/tmp`銆?- 浜х墿锛歚<scene>/I_000000.png + GT.png`锛坕nternal 鍙甫 R.png锛夈€乣pairs_manifest.json`銆乣FILE_INDEX.json`銆乣build_report.{json,md}`銆乣LOCK_bench_1108.json`銆?
## 3. 闃堝€间笌鍒ゅ畾琛紙棰勭櫥璁帮紱Lead 2026-11-08 鍐崇瓥宸插苟鍏ワ級
| 椤?| 闃堝€?| 鍒ゅ畾 |
|---|---|---|
| 鍑犱綍 shift (px) | 鈮?.0 | 瓒?鈫?REJECT锛堢‖绾㈢嚎锛?|
| 鍏夊害 g_affine | \|dev\|鈮?.05 | 瓒?鈫?REJECT锛坧rimary锛涢粦甯?GT 鍙ｅ緞锛?|
| 鍏夊害 g_med | \|dev\|鈮?.15 | 浠呰瘖鏂紱affine 姝ｅ父鑰?median 绾?鈫?WARN锛圛NSPECT_REFLECTION锛屾姭闇诧級 |
| 娈嬪樊 resid | 鈮?.2 / 1.2鈥?.8 / >1.8 | 姝ｅ父 / REVIEW / REJECT锛?*1.2 涓嶅彉**锛圓 缁勫亸楂樺睘棰勬湡锛孯EVIEW鈮犳嫆缁濓紝椤讳汉宸ョ瀛楋級 |
| 鍙嶅皠寮哄害 ref_abs_mean | <2.0 / 2鈥? / 4鈥?0 / >10 | too_weak鈫扲EVIEW锛泈eak/medium/strong 鍒嗙骇璁板綍 |
| cluster锛坲nion-find锛?| strict dHash鈮? / sameplace dHash鈮? | 涓ゆ。鍐?manifest/FILE_INDEX锛泂ameplace 涓哄彂甯?缁熻涓诲彛寰?|
| 鍦烘櫙绾?| affine 鏋佸樊>0.05 / GT 鍞竴鈮? / 閰嶅鏁?2鎴?8 | REVIEW / REVIEW / WARN |
| 榛戝竷鍏ョ敾 | 鏆?浣庣汗鐞嗚繛閫氬煙 area鈮?% 涓?border鈮?.22 涓?extent鈮?.60 | REVIEW锛堢洰妫€ overlay锛?|
| 娉勬紡 vs 鍙傝€冩睜 | md5 鍚屸啋REJECT锛沝Hash鈮? 鎴?corr鈮?.98鈫扲EVIEW锛沜orr鈮?.90鈫扺ARN | 瀵硅缁?娴嬭瘯姹犻兘瑕佽窇 |
| 闅愮锛堜汉鑴?杞︾墝锛?| Haar 鍊欓€夛紙渚ц劯/閬尅婕忔锛?| 鍙?REVIEW 涓嶈嚜鍔ㄦ嫆锛涘彂甯冨墠蹇呴』妯＄硦鍖?浜哄伐鐩 |
| 璁稿彲 | meta 缂哄け | REVIEW锛堥樆鏂彂甯冿級锛沗--allow-missing-license` 浠呭唴閮ㄦ睜 鈫?`INTERNAL_ONLY` |
| film_glass_spec | 瑙?搂10 | provenance only锛氱己澶变笉鎷掔粷锛屼絾 public 鎵规缂哄け 鈫?`release_gate.blocked` |

## 4. 鍥涙€佷笌閫€鍑虹爜
`PASS` 閫氳繃 锝?`WARN` 鍙叆搴撲絾鎶ュ憡椤绘姭闇?锝?`REVIEW` 浜哄伐澶嶆煡/绛惧瓧 锝?`REJECT` 涓嶅緱鍏ュ簱锛堝洖閫€璇ュ抚鎴栭噸鎷嶏級銆?qc/build 閫€鍑虹爜锛歚0` 鏃?REJECT/REVIEW銆乣2` 鏈?REJECT銆乣3` 鏈?REVIEW锛堟棤 REJECT锛夈€乣1` 閿欒/鏃犲悎鏍煎€欓€夛紱verify锛歚0` 涓€鑷淬€乣1` 缂哄け/绡℃敼銆?build 鍗充娇鎶婃墍鏈?REVIEW 鎺掗櫎锛屽彧瑕佹湰鎵?QC 璁℃暟鍚?REVIEW 浠?exit 3锛堟彁绀烘湁浜哄伐澶嶆煡椤癸級銆?
## 5. 鎵归噺鎶ュ憡锛圝SON+md锛屾瘡鎵逛竴浠斤紝绂佹瑕嗙洊鏃ф姤鍛婏級
- `qc_report.json`锛歚{schema, script_version, script_sha256, thresholds, batches:[{batch, root, layout, n_scenes, n_pairs, counts{PASS,WARN,REVIEW,REJECT,ERROR}, rows[], scenes[], content{I,GT}}]}`锛?  content 鍚屾椂缁?*璐績 first-seen** 涓?**union-find** 涓ゅ鍞竴鏁帮紙`n_unique_dhash2/8` vs `n_unique_dhash2_uf/8_uf`锛夈€?- `qc_report.md`锛毬? 鍙屽彛寰勮鏁?鈫?搂1 鎵圭骇璁℃暟 鈫?搂2 鍦烘櫙琛?鈫?搂3 閫愬琛?鈫?搂4 REJECT/REVIEW 娓呭崟鍚師鍥?鈫?搂5 鍐呭/鍞竴/娉勬紡 鈫?搂6 闅愮/璁稿彲 鈫?搂7 闄愬埗锛坄film_glass_spec` 鏄庣粏鍦?json 姣忚涓?build 浜х墿 summary 涓級銆?- build 浜х墿鍙﹀惈 `film_glass_spec_summary`锛坣_complete/partial/missing銆乥y_film_type銆乂LT 缁熻锛変笌 `release_gate`锛坆lockers 閫愭潯鍒楀嚭锛夈€?
## 6. 涓?`real_pair_nov02_1108` 鍏煎锛堢‖绾︽潫锛?- 甯冨眬锛歚<scene>/I_000000.png + GT.png`锛坕nternal 鍙甫 R.png锛夛紱鏍圭洰褰?`pairs_manifest.json`銆?- manifest 椤跺眰 11 閿笌 rows 22 閿叏淇濈暀锛涘彟鍔?v2 鎵╁睍 `qc_counts/privacy/license/selection/cluster_rule/n_clusters/cluster_groups/effective_n/
  usage/release_mode/r_mode/release_gate/film_glass_spec_summary/film_glass_spec_fields/n_spec_missing`锛?  rows 鍔?`cluster_id / cluster_id_sameplace / cluster_id_strict / n_clusters_* / review_accepted / r_package_path / r_provenance / film_glass_spec`銆?- `refl_grade` 淇濈暀 `"strong"/"weak"`锛坮ef_abs_mean 涓綅鏁颁簩鍒嗭級锛沗r_path` 闈炵┖锛沗abs_delta_frame/is_dev4/n_cand/omega_frac` 璇箟涓?1108 涓€鑷达紱dev4 鍥為€€ `DEV4_SCENES`銆?- **schema 鍏煎 鈮?閫夊抚闆嗙浉鍚?*锛歷2 鐢?affine primary锛堟瘮 1108 寤洪泦鏃?median g鈮?.15 鏇翠弗锛夛紱鍥炴斁 nov02 姹犻€夊抚瑙勫垯鍙鐜板師鍙戝竷闆嗭紙骞茶窇 10/10 甯у彿涓€鑷达級锛屼絾 QC 璁℃暟涓嶅悓锛屽紩鐢ㄩ』娉ㄦ槑銆?
## 7. 闅愮銆佽鍙€丷 涓庡彂甯冮棬锛堜笉鍥?QC PASS 鑰岃В闄わ級
- Haar 鍙綔 REVIEW 鎻愮ず锛坣ov02_1108 15 鍦烘櫙 0 鍛戒腑锛屼絾閬尅/渚ц劯浼氭紡妫€锛夆噿 **鍙戝竷鍓嶅己鍒?*锛氣憼 浜哄伐鐩 `privacy_flags/` overlay锛涒憽 浜鸿劯/杞︾墝妯＄硦鍖栵紱
  鈶?闅忔満 **10% 浜屾浜哄伐鎶芥煡**锛涗笁姝ュ潎鐣欑瀛楄褰曪紙Lead 鍐崇瓥 d锛夈€?- license 蹇呭～锛圠ead 鍐崇瓥 c锛夛細缂哄け榛樿 REVIEW 闃绘柇鍙戝竷锛沗--allow-missing-license` 浠呭唴閮ㄤ复鏃舵睜锛屼骇鐗?`usage=INTERNAL_ONLY`銆乣release_gate.blocked=true`锛屼笉寰楄繘鍏紑鍖咃紱public 鐩存帴绂佺敤銆?- **R 涓嶆槸 GT**锛圠ead 鍐崇瓥 2锛夛細鍏紑涓诲寘鍙斁 I / GT锛堥粦甯冿級/ 鎺╃爜鎴?meta / 璇存槑锛涢渶瑕?R 鍙兘璧?`aux/` 瀛愬寘骞跺甫 `estimated, not ground truth` + 鍏嶈矗 README銆?- `release_gate.blockers[]` 鍙栧€硷細`internal_only / license_missing_rows / privacy_flagged_rows / public_requires_film_glass_spec_rows /
  public_requires_manual_overlay_and_anonymization / public_requires_data_owner_cosign`锛汱OCK 鍙槸 QC 閿侊紝涓嶆槸娉曞緥 clearance銆?
## 8. cluster 涓庢湁鏁堢嫭绔嬫牱鏈紙union-find锛汱ead 2026-11-08 瀹氭锛?- 鍙屾。锛?*`cluster_sameplace`锛坉Hash鈮?锛? 鍙戝竷/缁熻涓诲彛寰?*锛?*`cluster_strict`锛坉Hash鈮?锛? 鏁忔劅鎬у彛寰?*锛涢檮 `n_clusters`銆乣cluster_groups`銆乣effective_n=sameplace 绨囨暟`銆?- nov02_1108 15 鍦烘櫙瀹炴祴锛坄qc_cluster_map_1108.py`锛屼笌鍙戝竷闆嗕竴鑷达級锛欸T dHash鈮? 鈫?**11 绨?*锛沝Hash鈮? 鈫?**5 绨?*锛? 涓?nov02a 瀹ゅ唴鍚岀皣 + 170120锛?  nov02b 3 鎴峰鍚岀皣 + 172611 + 172814锛夛紱I 涓?13/5 绨?鈬?**鏈夋晥鐙珛鏍锋湰鈮?锛堜富鍙ｅ緞锛?*銆傝瘉鎹細md5 鍏ㄤ笉鍚屻€?05 瀵规渶灏?GT MAD=3.66銆佹棤閲嶅瀵煎嚭銆?- 鈿狅笍 璁℃暟鍙ｅ緞锛氳椽蹇?first-seen锛圙T dHash2=12銆両 dHash2=14銆両 dHash8=6锛夆墵 union-find锛?1/13/5锛夛紝**涓嶅緱娣风敤**锛涙棫鎶ュ憡銆孖 dHash鈮? 浠?6 鍞竴銆嶅凡浣滃簾銆?- 缁熻褰掑睘锛圠ead 鍐崇瓥 3锛夛細QC/寤洪泦鎶ュ憡鍙斁**鎻忚堪鎬?*绨囪鏁颁笌绨囧唴鍧囧€硷紱**block bootstrap / CI / MDE 褰?exp-dev 鍗曚竴瀹炵幇**锛屾湰鑴氭湰涓嶆彁渚?CI銆?  璇勪及鎶ュ憡浠嶉』鎶ャ€屾枃浠舵暟 / 鍦烘櫙鏁?/ 鏈夋晥鐙珛鏍锋湰鏁?cluster)銆嶄笁琛屻€?
## 9. 鍦烘櫙绾у垝鍒嗕笌銆屾柟娉曞喕缁撳悗涓€娆℃€ц瘎浼般€嶉鐧昏妯℃澘
- 鍒掑垎锛欰 缁勶紙瀹ゅ唴锛? B 缁勶紙鎴峰琛楅亾锛夊悇鑷嫭绔嬬粺璁★紱dev4 绫讳技瑙掕壊=寮€鍙戦潰鏉匡紙鍙弽澶嶇湅锛夛紝鍏朵綑涓€娆℃€?test锛涚湅杩?GT 鐨勫満鏅笉寰楄繘 test銆?- 鍐荤粨鏃跺啓棰勭櫥璁帮細鈶?bench LOCK sha256锛涒憽 鍦烘櫙娓呭崟+A/B/dev4+cluster 鍒掑垎锛涒憿 涓绘寚鏍?PSNR + HF-PSNR/LapVar锛涒懀 蹇呭甫 identity+瀹樻柟鍐荤粨 ckpt锛?  鈶?鍒ゆ嵁闃堝€间笌鏂瑰悜锛涒懃 缁熻鍔熸晥锛坣=鍦烘櫙鏁帮紱n=15 鏃?MDE鈮?.26 dB锛涘惈 cluster 鏃舵寜绨囨暟閲嶇畻骞跺啓鏄庯級锛涒懄 涓€娆℃€у０鏄庯紱鈶?鎶ュ憡鍚屾椂缁?affine/median 鍙屽彛寰勮鏁般€?  鍘嗗彶璀﹀憡锛氬崟绉嶅瓙 <1 dB 鏁堝簲涓嶅彲淇★紙鑷傞棿绉嶅瓙 sd 0.5鈥?.0 dB锛夈€?
## 10. `film_glass_spec` 瀛楁缁勶紙benchmark schema锛汱ead task 2026-11-08锛?- 浣嶇疆锛歚meta.json` 鐨?`film_glass_spec`锛坰cene 绾ф垨 `film_glass_specs.<scene>`锛沺air entry 鍙鐩栵級锛宐uild 鍚庡啓鍏?manifest 姣忚涓?`film_glass_spec_summary`銆?- 瀛楁锛堜腑鑻辨槧灏勶級锛歚film_brand`(鑶滃搧鐗岋紝鍙┖) / `film_model`(鍨嬪彿锛屽彲绌? / `film_type`(鏌撹壊=dyed|閲戝睘=metallic|闄剁摲=ceramic|鏈煡=unknown) /
  `vlt_pct`(鍙鍏夐€忚繃鐜?) / `ir_rejection_pct`(IR闃婚殧%) / `uv_rejection_pct`(UV闃婚殧%) / `film_age_months`(璐磋啘鏃堕暱) /
  `glass_color`(鐜荤拑棰滆壊) / `glass_thickness_mm`(鍘氬害) / `is_curved`(鏇查潰) / `is_automotive_grade`(姹借溅绾? / `source`(spec_sheet|installer_record|visual_estimate|vin_decode|unknown)銆?- `spec_status`锛歚complete`=film_type 宸茬煡涓?VLT 鏈夋晥锛沗partial`=閮ㄥ垎瀛楁锛沗missing`=鍏ㄧ己銆傚潖鍊间涪寮冭繘 `spec_issues` 鈫?璇ュ WARN锛堜笉鎷掔粷锛夈€?- **鍙綔 provenance**锛氱己澶变笉鏀瑰彉 QC 鍒ゅ畾锛涗絾 **public 鎵规**缂哄け 鈫?`release_gate.blockers` 鍔?`public_requires_film_glass_spec_rows=N`銆傛湭鐭ュ啓 unknown锛岀姝㈢紪閫犮€?
## 11. real-car validity 閫氶亾锛坄data/validity/carwind_1108/`锛涘彧浣?validity study锛屼笉杩?benchmark锛?- 鐩殑锛氭妸銆屾垜浠殑璐磋啘鐜荤拑涓庣湡瀹炶溅绐楁€ц川鐩歌繎銆嶄粠缁忛獙瑙傚療鍗囩骇涓哄彲娴嬭瘉鎹紱閲囨牱鍗忚瑙?`2026.11.08_杞︾獥瀵圭収閲囨牱鍗忚_1108.md`锛堢敤鎴风収鍋氾級銆?- 鐩綍锛歚data/validity/carwind_1108/carwind_<car_id>/meta.json + pair_<k>/{I_*.png,GT.png}`锛沗meta.json` 鍚?car_id/film_glass_spec/glass_desc/lighting/camera/angle_estimate/captured_at/operator/license銆?- 鍛戒护锛歚$PY $V --mode selftest` / `--mode scan --root <dir>` / `--mode intake --root <dir> --out <dir>/_intake_1108` / `--mode verify --root <dir> --out <dir>/_intake_1108`銆?- 浜х墿锛歚validity_pairs.jsonl`锛堜竴琛?瀵癸紝QC-lite 鎸囨爣+film spec+`usage="validity-study-only"`锛夈€乣MANIFEST.json`锛堥€愭枃浠?sha256/n_cars/n_pairs/hints/errors/`not_for_benchmark=true`锛夈€乣REPORT.md`锛堣溅琛?閫愬琛?鐩 checklist锛夈€?- **涓嶅 benchmark QC 闃堝€?*锛堢湡瀹炶溅绐楀厜搴?娈嬪樊鍒嗗竷涓庡彈鎺ц缃笉鍚岋紝涓嶈兘娣锋瘮锛夛紱閲嶇偣鎻愮ず锛欸T 榛戝竷娈嬬暀锛坄gt_cloth.hit` 鈫?蹇呴』鐩锛夈€両 甯ч粦甯?閬尅鍏ョ敾銆侀殣绉佸€欓€夈€?- 閫€鍑虹爜锛歩ntake/scan `0` 姝ｅ父銆乣1` 鏈?errors锛堟枃浠剁己澶?璇讳笉浜嗭級锛泇erify `0` 涓€鑷淬€乣1` 缂哄け/绡℃敼銆傞敊鐨勬槸鏁版嵁灏遍噸鎷嶏紝涓嶈嚜鍔ㄦ敼闃堝€笺€?
## 12. 骞茶窇璇佹嵁锛堟湇鍔″櫒 CPU 鍙锛涙棩蹇?`/tmp/ri_dev/synthdev_1108/`锛?| 瀵硅薄 | 缁撴灉 |
|---|---|
| v1.2 鑷 | **55/55 PASS**锛坴1.1 47 椤?+ 8 椤癸細film spec complete/invalid/missing銆佺己澶变笉鏀瑰彉鍒ゅ畾銆乵anifest/summary/FILE_INDEX銆乸ublic spec blocker锛?|
| v1.2 validity 鑷 | **17/17 PASS**锛堝悎鎴?2 杞︼細g=1.10 妫€鍑恒€丟T 榛戝竷娈嬬暀鍛戒腑銆丣SONL/MANIFEST/REPORT 钀界洏銆乿erify 绡℃敼 exit1锛?|
| 鍙戝竷闆?15 鍦烘櫙 QC锛坴1.1 閫昏緫锛?| PASS0/WARN5/REVIEW10/REJECT0锛坋xit 3锛夛紱shift 0.04鈥?.04px銆乬_affine 0.954鈥?.003 鍏ㄨ繃锛汚 缁?resid 1.28鈥?.44 REVIEW銆丅 缁?1.01鈥?.08 閫氳繃 |
| internal build锛?0 鍦烘櫙/999 瀵癸紝绛惧瓧鐧藉悕鍗曪級 | 10/10 鍦烘櫙閫夊嚭涓诲锛?*n_dev4=4**锛沜ompat OK锛泂ameplace=2/strict=6銆乪ffective_n=2锛泆sage=INTERNAL_ONLY锛沴ock 31 files鈫抳erify OK |
| public build锛堟棤 owner_sign / 鏈?owner_sign锛?| 涓诲寘鏃?R銆乣aux/R/*.png` + 鍏嶈矗 README锛涙棤 co-sign鈫抌locked+cosign blocker锛涙湁 co-sign鈫抍osign 娑堝け浣嗕粛 blocked锛坙icense+鑴辨晱锛夛紱lock 32 files鈫抳erify OK |
| validity 骞茶窇锛坣ov02_1108 涓ゅ満鏅吉閫?2 杞︼級 | scan/intake/verify 鍏?exit 0锛汻EPORT.md 姝ｅ父锛圓: g_aff=0.993/resid 1.44锛汢: g_aff=1.016/resid 1.07锛?|
- 宸茬煡锛歳esid_warn=1.2 瀵?A 缁勭郴缁熸€у亸楂橈紙1.28鈥?.44锛夊睘棰勬湡锛?*涓嶆斁瀹?*锛圠ead 鍐崇瓥 a锛夛紱灏嗘潵鍒?A/B 闃堝€奸』鏂伴鐧昏 v3 涓斿湪閲囬泦鍓嶅畾銆?
## 13. 闄愬埗涓庡彉鏇?- 闄愬埗锛欻aar 闅愮鍙槸鎻愮ず锛涜鍙?娉曞緥鍚堣涓嶇敱鑴氭湰鍒ゅ畾锛汻 涓?8-bit 閲忓寲娲剧敓锛坮ef_abs 涓庢诞鐐圭暐宸?4.739 vs 4.574锛夛紱鏃ф睜鍥炴斁 affine primary 姣斿巻鍙?median 鍙ｅ緞澶氭嫆锛堥鏈燂級锛?  鑴氭湰鏈粡鐪熷疄 A/B/validity 鏁版嵁楠岃瘉锛屼粎 nov02 骞茶窇銆倂alidity 缁撴灉涓嶈兘涓?benchmark 鍒嗘暟娣锋帓锛堝垎甯冨墠鎻愪笉鍚岋級銆?- 鍙樻洿锛?*v1.2 (2026-11-08)**锛氭柊澧?`film_glass_spec` 瀛楁缁勶紙provenance锛宲ublic 缂哄け闃绘柇锛変笌 real-car validity 閫氶亾鑴氭湰 + 閲囨牱鍗忚锛泂elftest 47鈫?5銆佹柊澧?validity 17 椤广€?  **v1.1**锛氱櫧鍚嶅崟涓ユ牸 JSON+list_sha256+绛惧瓧鏉冮檺銆丷 涓嶈繘鍏紑鍖咃紙aux/ 鍏嶈矗锛夈€乧luster 缁熻褰?exp-dev銆?*v1.0**锛歜uild 榛樿 PASS/WARN銆乴icense 蹇呭～/INTERNAL_ONLY銆佸彂甯冮棬涓夋銆佷袱妗?union-find锛堟棫璐績浣滃簾锛夈€乺esid 1.2 涓嶅彉銆?  鍚庣画鏀归槇鍊?閫昏緫蹇呴』鍦ㄦ湰鑺傝鐗堟湰+鍘熷洜锛屼笖涓嶅緱鍦ㄦ寮忔暟鎹瘎浼板悗鏀瑰垽鎹€?- 鐜锛歚ssh better2`锛沗PY=<conda-env>/bin/python3`锛涜窇鍓?`export HOME=<home>; cd <project-root>`锛涗紶鑴氭湰涓嶅甫 CRLF銆?- 鍝堝笇锛?026-11-08 **v1.2**锛夛細`qc_bench_1108.py` sha256 `a66503afd436599250781f0f2d7a090e34f5c9c7b9049e129fb2de68010e6687`锛沗qc_cluster_map_1108.py` sha256 `7c06321aade2c091b713a75108c0187213704c0294ef096ab2a9b20777a8d150`锛?  `validity_carwind_1108.py` sha256 `9be63ea21f6793a1d659b92e72249ed7788d0b4a5b724489c4cd59f32f35f433`锛堟湰鍦?`code/local_tools/` 鈬?鏈嶅姟鍣?`scripts/` 涓€鑷达級銆?