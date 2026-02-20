# Predvidjanje nivoa depresije kod ljudi na osnovu audio zapisa

## Tim
- Veselin Roganović, SV 36/2022

## Asistent
- Dragan Vidaković

## Definicija problema
Predvidjanje nivoa depresije na [PHQ-8](https://selfmanagementresource.com/wp-content/uploads/English_-_PHQ-8-1.pdf) skali (skala ide od 0 do 24) na osnovu audio zapisa. Dodatno pojašnjenje, neće biti analiziran razgovor u smislu koje su reči izgovorene, već akcentovanje, intonacija, energija i generalno način govora.

## Motivacija
U današnjem svetu sa svim pritiscima, poredjenjima i problemima, sve više ljudi upada u depresiju. Procena da li je neko depresivan ili ne, često je zahtevan zadatak koji nema siguran DA/NE odgovor, čak i za iskusne psihoterapeute. S obzirom na to, korisno bi bilo napraviti model koji pomaže u tome, kao i istražiti kako bi izgledalo kada bi neki softver donosi tu odluku i koje osobine bi on analizirao (kao i gde najčešće greši i zašto).

## Skup podataka
[DAICWOZ](https://www.kaggle.com/datasets/saifzaman123445/daicwoz)

### Opis
Iz ovog dataset-a uzeo sam samo neophodne fajlove - audio zapise kliničkih intervjua i CSV fajlove koji ih ocenjuju (sadrže PHQ-8 metriku koja je za moje modele bitna). Ukupno bi se koristilo oko 140 audio zapisa intervjua i njihovih ocena.

## Metodologija
Skup podataka biće podeljen na trening, validacioni i test skup u odnosu 80:10:10. Audio zapisi biće prethodno procesirani (resampling, normalizacija, uklanjanje tišine), nakon čega će biti izdvojene akustičke karakteristike poput MFCC koeficijenata, pitch-a i energije.

Biće primenjena i uporedjena dva pristupa:
1. Klasični pristup zasnovan na akustičkim obeležjima i Support Vector Regression (SVR) modelu.
2. Model dubokog učenja zasnovan na CNN + LSTM arhitekturi nad MFCC matricama radi predviđanja PHQ-8 skora.

## Evaluacija
Model predviđa PHQ-8 skor, koji je ceo broj od 0 do 24. Zbog toga, koristio bih MAE, RMSE i Pearson correlation coefficient kao metrike za evaluaciju. 

## Linkovi do par radova na sličnu temu

- [Audio depression recognition based on deep learning](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13553/1355306/Audio-depression-recognition-based-on-deep-learning/10.1117/12.3059130.full?tab=ArticleLink)
- [Detecting Depression with Audio/Text Sequence Modeling of Interviews](https://sls.csail.mit.edu/publications/2018/Alhanai_Interspeech-2018.pdf)
- [A review of depression and suicide risk assessment using speech analysis](https://www.sciencedirect.com/science/article/abs/pii/S0167639315000369)
