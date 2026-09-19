"""Paths, voices and test sentences shared by every stage, so PyTorch and llama.cpp get identical inputs."""
W = "/root/work"
OUT_ASR = f"{W}/out/asr"
OUT_TTS = f"{W}/out/tts"
TTS_NAME = "qwen3-tts-1.7b-kreyol"
VOICES = ["kreyol_f1", "kreyol_f2", "kreyol_f3", "kreyol_m1", "kreyol_v5"]
EVAL_VOICES = ["kreyol_f1", "kreyol_m1"]

# Read by each named voice of the -voices model to make its reference clip for cloning.
REF_TEXT = "Bonjou, mwen se yon vwa kreyòl. Mwen ka li nenpòt tèks ou ban mwen, byen klè e san prese."

SENTENCES = [
    "Bonjou, kijan ou ye jodi a? Mwen espere tout bagay ap mache byen pou ou.",
    "Lekòl la ap louvri pòt li lendi pwochen a 8:30 nan maten.",
    "Si w bezwen plis enfòmasyon, tanpri rele biwo nou an oswa voye yon mesaj.",
    "Lapli a tonbe tout lannwit, men solèy la leve bèl bonè maten an.",
    "Pitit la te kontan anpil lè manman l pote yon bèl gato pou fèt li.",
    "Nou dwe pwoteje anviwònman an pou jenerasyon k ap vini yo.",
    "Doktè a di m pou m pran medikaman an de fwa pa jou pandan yon semèn.",
    "Machann nan vann twa liv diri pou 500 goud nan mache a.",
    "Ki lè bis la ap pase? Mwen pa vle rive an reta nan travay mwen.",
    "Ayiti se yon peyi ki gen anpil istwa, kilti ak bèl mizik.",
    "Timoun yo ap jwe foutbòl nan lakou lekòl la apre klas.",
    "Mèsi anpil pou èd ou, mwen pap janm bliye sa w fè pou mwen.",
]

# 24 more for the larger comparison: with 12 sentences per voice, one run's sampling noise is about a CER
# point, the size of every difference between variants. BIG=1 uses all 36 with fresh seeds.
SENTENCES_EXTRA = [
    "Mwen renmen koute radyo nan maten lè m ap prepare kafe.",
    "Fanmi an ap reyini lakay grann nan pou Nwèl ane sa a.",
    "Tanpri, fèmen fenèt la paske van an ap soufle twòp.",
    "Pwofesè a mande elèv yo pou yo li chapit sa a anvan jedi.",
    "Nou bezwen achte pen, lèt ak ze nan boutik la.",
    "Kòman w ap fè pou w rive Okap si pa gen machin?",
    "Lè lapli tonbe, lari yo plen dlo e trafik la bloke.",
    "Li travay kòm enfimyè nan lopital la depi dis lane.",
    "Pa bliye pran parapli w, syèl la sanble l pral kouvri.",
    "Jodi a se jou fèt endepandans peyi a, tout moun an fèt.",
    "Mwen te pèdi kle machin mwen yè swa, men m jwenn yo maten an.",
    "Chak dimanch, nou ale legliz epi nou manje ansanm apre.",
    "Ti gason an te kouri vit pou l te ka trape bis la.",
    "Mango yo mi, n ap kapab fè ji pou timoun yo.",
    "Ou konnen ki kote m ka jwenn yon famasi ki louvri kounye a?",
    "Agrikiltè yo ap tann lapli pou yo ka plante mayi ak pwa.",
    "Konsè a kòmanse a sèt è, men pòt yo ap louvri pi bonè.",
    "Mwen pa t konprann sa l te di a, li te pale twò vit.",
    "Lanmè a te kalm, pechè yo te rale anpil pwason jodi a.",
    "Si w vle aprann yon lang, fòk ou pratike l chak jou.",
    "Manman m fè yon bon soup joumou chak premye janvye.",
    "Elektrisite a koupe ankò, nou pral limen yon lanp.",
    "Kilè w ap vin vizite nou? Nou poko wè w depi lontan.",
    "Nou swete w yon bon vwayaj ak anpil siksè nan travay ou.",
]

# Held-out CMU test clips shipped as the Space's examples (CMU licence: notice kept in the Space repo).
CMU_EXAMPLES = {
    "cmu_apre": "Aprè sa, li lave figi l, li sòti, li kenbe pou li pa kriye ankò, epi li bay lòd sèvi manje a.",
    "cmu_legliz": "Men, pou legliz an Ayiti yo, yo pa sèvi ak tanbou nan sèvis yo paske sa sanble ak yon selebrasyon vodou.",
    "cmu_chwal": "Nou mete yon mò nan bouch chwal pou fè yo obeyi nou.",
    "cmu_diven": "li te fè moun tout nasyon yo bwè nan diven l lan ki fò anpil, li fè yo bwè diven gwo imoralite li a.",
}
