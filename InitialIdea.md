# Predvidjanje nivoa depresije kod ljudi na osnovu audio zapisa

## Tim
- Veselin Roganović, SV 36/2022

## Asistent
- Dragan Vidaković

## Definicija problema i motivacija
U današnjem svetu sa svim pritiscima, poredjenjima i problemima, sve više ljudi upada u depresiju. Procena da li je neko depresivan ili ne, često je zahtevan zadatak koji nema siguran DA/NE odgovor, čak i za iskusne psihoterapeute. S obzirom na to, korisno bi bilo istražiti kako bi izgledalo kada bi neki softver donosi tu odluku i koje osobine bi on analizirao (kao i gde najčešće greši i zašto).

Naravno, nikakav softver ne može dati 100% tačan odgovor u ovakvoj situaciji, već može samo sugerisati. Uz to, moj cilj sa ovim projektom nije da napravim model koji bi direktno bio korišćen u nekom širem softveru (za sada), već da napravim dobar model, te analiziram i istražim kako on donosi zaključke, tj. koji faktori najviše utiču na procenu da li je neko depresivan ili ne.

U ovom projektu, baviću se isključivo procenom na osnovu audio zapisa. Preciznije rečeno, analiziraće se tonalitet, način govora, akcentovanje i slično. Sam sadržaj razgovora neće biti predmet analize. Kao proširenje u budućnosti, planiram da dodam i ekstrakciju teksta iz audio zapisa, kao i analizu samog teksta ili dijaloga. Dodatno, u budućnosti možda ubacim i analizu videa.

Dakle, šta konkretno ću rešavati ovde? Na osnovu audio zapisa cilj je proceniti nivo depresije kod osobe koja govori. Postoje razne skale za ovo, a ja planiram koristiti jednu vrlo popularnu - [PHQ-8](https://selfmanagementresource.com/wp-content/uploads/English_-_PHQ-8-1.pdf).

Vrlo je bitno takodje naglasiti da planiram da isprobam više metodologija i uporedim ih, kako po evaluaciji, tako i po analizi samih rezultata i zaključaka svakog modela.

## Skup podataka
[Extended DAIS](https://dcapswoz.ict.usc.edu/) (poslao sam zahtev za pristup i čekam odobrenje)

### Opis (dok ne dobijem pristup samo je pretpostavka iz svega što sam pročitao)
Extended DAIC Database je istraživačka baza podataka za detekciju depresije koja sadrži audio snimke govora, tekstualne transkripte i video snimke lica ispitanika tokom intervjua. Svaki ispitanik ima kliničku anotaciju (PHQ-8 skor), što omogućava nadgledano učenje. Baza je relativno mala (275 intervjuisanih učesnika), ali visokog kvaliteta i često se koristi za audio, NLP i multimodalne modele u mentalnom zdravlju.

## Metodologija
Plan je da se isproba i uporedi više metodologija i postoje šanse da će se one razlikovati od onih koje ću opisati ovde, a to su:
- Feature extraction (postoje šanse da su već ekstraktovani u nekom fajlu dataset-a) + SVR
- CNN + LSTM (RNN)
- WavLM (pretrained transformer) + LSTM

Takodje, pretprocesiranje podataka biće neophodno, ali s obzirom da još nemam pristup podacima, ovo ne mogu komentarisati.

## Evaluacija
Planiram koristiti podatke kao unapred podeljene na train/val/test. Pretpostavljam da je to već učinjeno u skupu podataka (zbog nečega što sam pročitao), ali ukoliko nije, sam ću podeliti.

Model predviđa PHQ-8 skor, koji je ceo broj od 0 do 24. Zbog toga, koristio bih MAE, RMSE i Pearson correlation coefficient kao metrike za evaluaciju i poredjenje. 

## Linkovi do par radova na sličnu temu

- [Audio depression recognition based on deep learning](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13553/1355306/Audio-depression-recognition-based-on-deep-learning/10.1117/12.3059130.full?tab=ArticleLink)
- [Detecting Depression with Audio/Text Sequence Modeling of Interviews](https://sls.csail.mit.edu/publications/2018/Alhanai_Interspeech-2018.pdf)
- [A review of depression and suicide risk assessment using speech analysis](https://www.sciencedirect.com/science/article/abs/pii/S0167639315000369)
