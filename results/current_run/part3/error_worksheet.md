# Error-analysis worksheet (Part 3)

Final classifier: fact recall>=0.30, threshold 0.2535. Dev accuracy 193/221 = 87.33%; 10 false positives, 18 false negatives. Seeded random selection (seed 0): 10 FP and 10 FN marked SELECTED.

Probabilities are E(ntailment) / N(eutral) / C(ontradiction). A pruned fact never reached DeBERTa in the final run; its evidence is shown to judge whether pruning caused the error.

## False positives (predicted S, gold NS or IR)

### FP01 (SELECTED): Cleveland is in Ohio.

- gold IR, predicted S, score 0.9936, word overlap 1.000
- evidence 1 (E 0.994 / N 0.006 / C 0.001): Trivisonno was born in East Cleveland, Ohio, and attended Mayfield High School in
- evidence 2 (E 0.993 / N 0.007 / C 0.001): Personal life. Trivisonno was born in East Cleveland, Ohio, and attended Mayfield High School in
- evidence 3 (E 0.753 / N 0.246 / C 0.001): Cleveland's eastern suburbs. He was married to his high school sweetheart Linda Conforto, who died of lung and brain cancer in 2009, at age 61.
- most contradicting (E 0.028 / N 0.735 / C 0.237): His son, Michael Jr., owns an upscale restaurant called Trivs in suburban Strongsville, Ohio.
- article lead sentence: Mike Trivisonno Mike Trivisonno (September 20, 1947 – October 28, 2021) was an American radio broadcaster, known for hosting his own self-titled afternoon talk show on Cleveland radio station WTAM AM 1100.
- category: 
- notes: 

### FP02 (SELECTED): She transitioned to acting.

- gold NS, predicted S, score 0.8121, word overlap 0.500
- evidence 1 (E 0.812 / N 0.160 / C 0.028): Virginia Valli Virginia Valli (died September 24, 1968) was an American stage and film actress whose motion picture career started in the silent film era and lasted until the beginning of the sound film era of the 1930s.
- evidence 2 (E 0.797 / N 0.201 / C 0.002): Valli performed in "The Man Who Found Himself" with Thomas Meighan. Her first sound picture was "The Isle of Lost Ships" with Jason Robards Sr.
- evidence 3 (E 0.376 / N 0.561 / C 0.063): Early life. Born Virginia McSweeney in Chicago, Illinois, she got her acting start in Milwaukee with a stock company.
- most contradicting (E 0.002 / N 0.003 / C 0.994): Her last film was in "Night Life in Reno", in 1931.
- article lead sentence: Virginia Valli Virginia Valli (died September 24, 1968) was an American stage and film actress whose motion picture career started in the silent film era and lasted until the beginning of the sound film era of the 1930s.
- category: 
- notes: 

### FP03 (SELECTED): Tusa has worked to promote Palermo as a cultural destination.

- gold IR, predicted S, score 0.2853, word overlap 0.400
- evidence 1 (E 0.285 / N 0.637 / C 0.078): References.
- evidence 2 (E 0.133 / N 0.827 / C 0.040): Biography.
- evidence 3 (E 0.123 / N 0.849 / C 0.028): In 1947.
- most contradicting (E 0.010 / N 0.860 / C 0.129): Tusa promoted excavations in the archaeological sites of Soluntum, Segesta, Selinunte, Motya, Marsala.
- article lead sentence: Vincenzo Tusa Vincenzo Tusa (12 July 1920 – 5 March 2009) was an Italian archeologist.
- category: 
- notes: 

### FP04 (SELECTED): Itakura has represented Japan at the Under-19-level.

- gold NS, predicted S, score 0.5125, word overlap 0.500
- evidence 1 (E 0.513 / N 0.451 / C 0.037): In May 2017, Itakura was called up to the Japan U-20 national team for the 2017 U-20 World Cup. At this tournament, he played 2 matches as defensive midfielder.
- evidence 2 (E 0.360 / N 0.438 / C 0.202): International career. In May 2017, Itakura was called up to the Japan U-20 national team for the 2017 U-20 World Cup.
- evidence 3 (E 0.322 / N 0.638 / C 0.040): In May 2017, Itakura was called up to the Japan U-20 national team for the 2017 U-20 World Cup.
- most contradicting (E 0.004 / N 0.767 / C 0.230): He was selected for the senior Japan
- article lead sentence: Ko Itakura
- category: 
- notes: 

### FP05 (SELECTED): Trivisonno worked at WMMS.

- gold NS, predicted S, score 0.9650, word overlap 0.500
- evidence 1 (E 0.965 / N 0.008 / C 0.027): He was a frequent caller on the "Sportsline" radio show on WTAM AM 1100 (during this period the station was operating under the WWWE call letters). The show was hosted by the acerbic Pete Franklin and dealt with Cleveland sports.
- evidence 2 (E 0.213 / N 0.031 / C 0.756): Mike Trivisonno Mike Trivisonno (September 20, 1947 – October 28, 2021) was an American radio broadcaster, known for hosting his own self-titled afternoon talk show on Cleveland radio station WTAM AM 1100.
- evidence 3 (E 0.106 / N 0.011 / C 0.883): Mike Trivisonno Mike Trivisonno (September 20, 1947 – October 28, 2021) was an American radio broadcaster, known for hosting his own self-titled afternoon talk show on Cleveland radio station WTAM AM 1100. He was widely referred to simply as "Triv" by callers and the Cleveland media.
- most contradicting (E 0.000 / N 0.001 / C 0.999): Trivisonno eventually began his own broadcasting career in 1987, working for a time at WNCX, commenting on sports (particularly boxing) and becoming a full-time personality from 1989–1992. He began hosting "Sportsline" on WWWE in 1994.
- article lead sentence: Mike Trivisonno Mike Trivisonno (September 20, 1947 – October 28, 2021) was an American radio broadcaster, known for hosting his own self-titled afternoon talk show on Cleveland radio station WTAM AM 1100.
- category: 
- notes: 

### FP06 (SELECTED): Trivisonno worked at WKNR.

- gold NS, predicted S, score 0.9929, word overlap 1.000
- evidence 1 (E 0.993 / N 0.003 / C 0.004): in the ratings, even though it was usually up against Cleveland Indians games on WKNR. It did so well, the decision was made to put Trivisonno in afternoon drive-time (under the "Sportsline" banner until 1998; under his own name since) ultimately leaving "Sportsline" to other hosts.
- evidence 2 (E 0.039 / N 0.397 / C 0.564): Personal life.
- evidence 3 (E 0.033 / N 0.962 / C 0.006): Cleveland's eastern suburbs.
- most contradicting (E 0.000 / N 0.001 / C 0.999): His brother, Gary, a golfer, was also a radio personality on WTAM, hosting a weekly golf show.
- article lead sentence: Mike Trivisonno Mike Trivisonno (September 20, 1947 – October 28, 2021) was an American radio broadcaster, known for hosting his own self-titled afternoon talk show on Cleveland radio station WTAM AM 1100.
- category: 
- notes: 

### FP07 (SELECTED): She has worked with a number of organizations.

- gold NS, predicted S, score 0.9243, word overlap 0.333
- evidence 1 (E 0.924 / N 0.075 / C 0.001): She also delivers awareness raising sessions to doctors and the Police to assist in their understanding of FGM. Her testimonies have appeared in numerous publications, including the BBC, the "Guardian" and the "Telegraph".
- evidence 2 (E 0.871 / N 0.127 / C 0.002): She currently resides in Walthamstow, London, where she worked as a mediator and a regular FGM educator for Waltham Forest Borough. Her testimonials and campaigning work have made her one of Britain's most prominent campaigners about FGM and she has appeared in numerous publications, including the "Telegraph", the BBC, and "The Guardian".
- evidence 3 (E 0.656 / N 0.342 / C 0.002): Wardere has worked as a mediator and FGM educator since then, helping young students escape FGM. She also delivers awareness raising sessions to doctors and the Police to assist in their understanding of FGM.
- most contradicting (E 0.055 / N 0.874 / C 0.071): Personal life.
- article lead sentence: Hibo Wardere Hibo Wardere is a Somali-born campaigner against female genital mutilation (FGM), author, and public speaker.
- category: 
- notes: 

### FP08 (SELECTED): Mayo Clinic is an organization.

- gold NS, predicted S, score 0.8417, word overlap 0.667
- evidence 1 (E 0.842 / N 0.153 / C 0.005): Early life and education. Oliphant has a Ph.D. in Biomedical Engineering from the Mayo Clinic and B.S. and M.S. degrees in Mathematics and Electrical Engineering from Brigham Young University.
- evidence 2 (E 0.817 / N 0.178 / C 0.005): Oliphant has a Ph.D. in Biomedical Engineering from the Mayo Clinic and B.S. and M.S. degrees in Mathematics and Electrical Engineering from Brigham Young University.
- evidence 3 (E 0.706 / N 0.283 / C 0.011): feasibility of designing a high-level data-parallel language extension to Python on graphics processing units (GPUs). On January 1, 2018, Oliphant announced he was leaving Anaconda Inc. He subsequently co-founded Quansight later that same year.
- most contradicting (E 0.011 / N 0.151 / C 0.838): He is also a member of the advisory council of the non-profit scientific computing foundation NumFOCUS.
- article lead sentence: Travis Oliphant Travis Oliphant is an American data scientist and businessman.
- category: 
- notes: 

### FP09 (SELECTED): He served as a sniper.

- gold NS, predicted S, score 0.9988, word overlap 1.000
- evidence 1 (E 0.999 / N 0.001 / C 0.000): Rob Furlong Rob Furlong (born 11 November 1976) is a Canadian former military sniper who, from March 2002 until November 2009, held the world record for the longest confirmed sniper kill in combat, at. His record stood for over 7 years and was improved upon by Craig Harrison with a distance of using a L115A3 Long Range Rifle.
- evidence 2 (E 0.997 / N 0.002 / C 0.000): Rob Furlong Rob Furlong (born 11 November 1976) is a Canadian former military sniper who, from March 2002 until November 2009, held the world record for the longest confirmed sniper kill in combat, at.
- evidence 3 (E 0.992 / N 0.007 / C 0.001): -range sniper weapon (LRSW), a.50-calibre McMillan Brothers Tac-50 rifle, loaded with Hornady A-MAX 750 gr very-low-drag bullets. He began firing at a fighter carrying an RPK machine gun.
- most contradicting (E 0.000 / N 0.002 / C 0.997): Furlong was stationed in Bosnia in 1999 as a peace keeper.
- article lead sentence: Rob Furlong Rob Furlong (born 11 November 1976) is a Canadian former military sniper who, from March 2002 until November 2009, held the world record for the longest confirmed sniper kill in combat, at.
- category: 
- notes: 

### FP10 (SELECTED): Jidenna's music is a fusion of hip-hop.

- gold NS, predicted S, score 0.8573, word overlap 0.750
- evidence 1 (E 0.857 / N 0.140 / C 0.002): His song "Classic Man" was nominated for Best Rap/Sung Collaboration at the 58th Grammy Awards. In June of the same year, Jidenna performed the song with Monáe at the BET Awards.
- evidence 2 (E 0.648 / N 0.349 / C 0.003): EP was released – "Yoga" by Janelle Monáe and Jidenna. His song "Classic Man" was nominated for Best Rap/Sung Collaboration at the 58th Grammy Awards.
- evidence 3 (E 0.479 / N 0.516 / C 0.006): His song "Classic Man" was nominated for Best Rap/Sung Collaboration at the 58th Grammy Awards.
- most contradicting (E 0.001 / N 0.196 / C 0.803): However, he would not adopt his signature dandy style until the death of his father in 2010. Jidenna describes his look as "heavily inspired by the Harlem Renaissance with hints of traditional West African design," and a "marriage of European and African aesthetics."
- article lead sentence: Jidenna Jidenna Theodore Mobisson (born May 4, 1985), known mononymously as Jidenna, is an American rapper and singer.
- category: 
- notes: 

## False negatives (predicted NS, gold S)

### FN01: Gerhard Fischer is best known for inventing a metal detector.

- gold S, predicted NS, score 0.0109, word overlap 0.600
- evidence 1 (E 0.011 / N 0.988 / C 0.001): Fisher shared the idea with Albert Einstein who correctly predicted the proliferation of hand held metal detector use. In 1937, he patented his metal detector.
- evidence 2 (E 0.006 / N 0.993 / C 0.001): Gerhard Fischer (inventor) Gerhard Julius Richard Fischer later Gerhard Julius Richard Fisher, Gerhard R. Fisher (March 18, 1899 – May 2, 1988) was a U.S. entrepreneur of German descent. He contributed to the development and popularity of the hand held metal detector.
- evidence 3 (E 0.006 / N 0.989 / C 0.005): References. - - -
- most contradicting (E 0.000 / N 0.002 / C 0.998): A metal detector had been invented some forty years earlier (1881) by Alexander Graham Bell for the sole purpose of locating a lead bullet in President James A. Garfield.
- article lead sentence: Gerhard Fischer (inventor) Gerhard Julius Richard Fischer later Gerhard Julius Richard Fisher, Gerhard R. Fisher (March 18, 1899 – May 2, 1988) was a U.S. entrepreneur of German descent.
- category: 
- notes: 

### FN02 (SELECTED): Jean Daullé was known for his portraits.

- gold S, predicted NS, score 0.1472, word overlap 0.500
- evidence 1 (E 0.147 / N 0.849 / C 0.003): Work. He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time.
- evidence 2 (E 0.145 / N 0.852 / C 0.003): He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time.
- evidence 3 (E 0.100 / N 0.897 / C 0.002): He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time. He marked his works J. D. The following are his principal plates:
- most contradicting (E 0.004 / N 0.276 / C 0.720): He then went to Paris, and worked at the studios of Robert Hecquet (1693-1775), who was also originally from Picardy.
- article lead sentence: Jean Daullé Jean Daullé (18 May 1703 – 23 April 1763) was a French engraver.
- category: 
- notes: 

### FN03 (SELECTED): The Taker trilogy is a novel by Katsu.

- gold S, predicted NS, score 0.1131, word overlap 1.000
- evidence 1 (E 0.113 / N 0.693 / C 0.194): Personal life.
- evidence 2 (E 0.079 / N 0.730 / C 0.190): Career.
- evidence 3 (E 0.041 / N 0.535 / C 0.423): Biography.
- most contradicting (E 0.002 / N 0.421 / C 0.576): The second novel, "The Reckoning", was nominated for several year-end awards including Goodreads Readers Choice Award for best paranormal fantasy and RT Book Reviews Reviewers Choice Award for best paranormal romance. " The Deep" was a finalist for the Bram Stoker Award for Superior Achievement in a Novel and a finalist for the Locus Award for Best Horror Novel.
- article lead sentence: Alma Katsu Alma Katsu (born 1959) is an American writer of adult fiction.
- category: 
- notes: 

### FN04: The Hunger is a novel by Katsu.

- gold S, predicted NS, score 0.0739, word overlap 1.000
- evidence 1 (E 0.074 / N 0.611 / C 0.315): Personal life.
- evidence 2 (E 0.020 / N 0.341 / C 0.639): Biography.
- evidence 3 (E 0.015 / N 0.980 / C 0.005): Bibliography.
- most contradicting (E 0.001 / N 0.003 / C 0.996): She published her first novel, "The Taker," in 2011 through Gallery Books.
- article lead sentence: Alma Katsu Alma Katsu (born 1959) is an American writer of adult fiction.
- category: 
- notes: 

### FN05 (SELECTED): The Deep is a novel by Katsu.

- gold S, predicted NS, score 0.0739, word overlap 1.000
- evidence 1 (E 0.074 / N 0.564 / C 0.362): Personal life.
- evidence 2 (E 0.027 / N 0.963 / C 0.010): Bibliography.
- evidence 3 (E 0.014 / N 0.207 / C 0.779): Biography.
- most contradicting (E 0.001 / N 0.003 / C 0.996): She published her first novel, "The Taker," in 2011 through Gallery Books.
- article lead sentence: Alma Katsu Alma Katsu (born 1959) is an American writer of adult fiction.
- category: 
- notes: 

### FN06: Red Widow is a novel by Katsu.

- gold S, predicted NS, score 0.0923, word overlap 1.000
- evidence 1 (E 0.092 / N 0.663 / C 0.245): Personal life.
- evidence 2 (E 0.060 / N 0.304 / C 0.636): Career.
- evidence 3 (E 0.038 / N 0.950 / C 0.013): Bibliography.
- most contradicting (E 0.000 / N 0.002 / C 0.998): She published her first novel, "The Taker," in 2011 through Gallery Books.
- article lead sentence: Alma Katsu Alma Katsu (born 1959) is an American writer of adult fiction.
- category: 
- notes: 

### FN07 (SELECTED): Hibo Wardere is based in the United Kingdom.

- gold S, predicted NS, score 0.0000, word overlap 0.000, PRUNED before DeBERTa
- evidence 1 (E 0.999 / N 0.001 / C 0.001): Born in Somalia, she moved to London, England when just a teenager in 1989, as a refugee fleeing the Somali Civil War. She currently resides in Walthamstow, London, where she worked as a mediator and a regular FGM educator for Waltham Forest Borough.
- evidence 2 (E 0.997 / N 0.002 / C 0.001): She currently resides in Walthamstow, London, where she worked as a mediator and a regular FGM educator for Waltham Forest Borough. Her testimonials and campaigning work have made her one of Britain's most prominent campaigners about FGM and she has appeared in numerous publications, including the "Telegraph", the BBC, and "The Guardian".
- evidence 3 (E 0.995 / N 0.004 / C 0.001): Hibo Wardere Hibo Wardere is a Somali-born campaigner against female genital mutilation (FGM), author, and public speaker. Born in Somalia, she moved to London, England when just a teenager in 1989, as a refugee fleeing the Somali Civil War.
- most contradicting (E 0.000 / N 0.011 / C 0.989): Early life. Hibo Wardere was born in Somalia.
- article lead sentence: Hibo Wardere Hibo Wardere is a Somali-born campaigner against female genital mutilation (FGM), author, and public speaker.
- category: 
- notes: 

### FN08: She was born on July 18, 1987.

- gold S, predicted NS, score 0.0012, word overlap 0.500
- evidence 1 (E 0.001 / N 0.996 / C 0.002): During the Miss International 2011 final, Jessica Bar
- evidence 2 (E 0.001 / N 0.997 / C 0.002): Personal life.
- evidence 3 (E 0.001 / N 0.998 / C 0.001): Miss International. She represented Venezuela in the 2011 Miss International pageant in Chengdu, China in November 6, 2011.
- most contradicting (E 0.000 / N 0.000 / C 0.999): Jessica Barboza Jessica Cristina Barboza Schmidt (born in Maracaibo, Venezuela on 14 August 1987) is a Venezuelan model and beauty pageant titleholder who won the titles of Miss Earth Venezuela 2009 and Miss International Venezuela 2010.
- article lead sentence: Jessica Barboza Jessica Cristina Barboza Schmidt (born in Maracaibo, Venezuela on 14 August 1987) is a Venezuelan model and beauty pageant titleholder who won the titles of Miss Earth Venezuela 2009 and Miss International Venezuela 2010.
- category: 
- notes: 

### FN09 (SELECTED): With CSKA, he won domestic titles.

- gold S, predicted NS, score 0.0350, word overlap 0.500
- evidence 1 (E 0.035 / N 0.914 / C 0.051): ation and the minimum fee release clause.
- evidence 2 (E 0.017 / N 0.972 / C 0.010): Honours.:Club. Internacional - Campeonato Gaúcho: 2002, 2003 - Copa Sudamericana: 2008 CSKA Moscow - Russian Premier League: 2005, 2006 - Russian Cup: 2004–05, 2005–06, 2008–09 - Russian Super Cup: 2004, 2006, 2007, 2009 - UEFA Cup: 2004–05 Palmeiras - Copa do Brasil: 2012 Botafogo - Campeonato Brasileiro Série B: 2015 Goiás - Campeonato Goiano: 2016
- evidence 3 (E 0.014 / N 0.977 / C 0.009): Internacional - Campeonato Gaúcho: 2002, 2003 - Copa Sudamericana: 2008 CSKA Moscow - Russian Premier League: 2005, 2006 - Russian Cup: 2004–05, 2005–06, 2008–09 - Russian Super Cup: 2004, 2006, 2007, 2009 - UEFA Cup: 2004–05 Palmeiras - Copa do Brasil: 2012 Botafogo - Campeonato Brasileiro Série B: 2015 Goiás - Campeonato Goiano: 2016
- most contradicting (E 0.000 / N 0.003 / C 0.997): Career.:International career.
- article lead sentence: Daniel Carvalho Daniel da Silva Carvalho (born 1 March 1983), more commonly known as Daniel Carvalho, is a Brazilian former football attacking midfielder.
- category: 
- notes: 

### FN10 (SELECTED): Florencia Bertotti is an Argentinian.

- gold S, predicted NS, score 0.0000, word overlap 0.000, PRUNED before DeBERTa
- evidence 1 (E 0.978 / N 0.013 / C 0.009): Florencia Bertotti María Florencia Bertotti (born 15 March 1983), better known as Florencia Bertotti is an Argentine actress, singer, songwriter and businesswoman.
- evidence 2 (E 0.731 / N 0.264 / C 0.005): Her father passed away in 1999 when she was filming the series "Verano del '98". Florencia studied at the Colegio Nuestra Señora de la Misericordia in Recoleta, Buenos Aires, Argentina.
- evidence 3 (E 0.068 / N 0.927 / C 0.005): Florencia studied at the Colegio Nuestra Señora de la Misericordia in Recoleta, Buenos Aires, Argentina.
- most contradicting (E 0.001 / N 0.782 / C 0.217): In 2001, she was part of the cast of the television series "Culpables". In 2001, she acted in the movie "Déjala correr".
- article lead sentence: Florencia Bertotti María Florencia Bertotti (born 15 March 1983), better known as Florencia Bertotti is an Argentine actress, singer, songwriter and businesswoman.
- category: 
- notes: 

### FN11: Jonathan Haagensen was born in Rio de Janeiro.

- gold S, predicted NS, score 0.0532, word overlap 0.500
- evidence 1 (E 0.053 / N 0.800 / C 0.147): iscient"
- evidence 2 (E 0.008 / N 0.980 / C 0.013): ro".
- evidence 3 (E 0.005 / N 0.992 / C 0.003): Discography. - (
- most contradicting (E 0.000 / N 0.005 / C 0.995): Life and career. A resident of the Vidigal community since his birth, he has Norwegian ancestry from his father, who has not seen since he was six years old. " I never knew anything about my father. I only know that he lives in Vitória, the capital city of the state of Espírito Santo," he says.
- article lead sentence: Jonathan Haagensen Jonathan Haagensen Cerqueira (born February 23, 1983) is a Brazilian actor, model, and singer.
- category: 
- notes: 

### FN12: Rio de Janeiro is in Brazil.

- gold S, predicted NS, score 0.0000, word overlap 0.250, PRUNED before DeBERTa
- evidence 1 (E 0.454 / N 0.544 / C 0.002): 2000 - "Palace II" - Short - (Fernando Meirelles and Kátia Lund) - 2002 - "City of God (Cidade de Deus)" - (Fernando Meirelles and Kátia Lund) - 2002 - "Seja o que Deus Quiser" - (Murilo Salles) - 2004 - "Four for None (O Diabo a Quatro)" - (Alice de Andrade) - 2002/05 - "City of Men (Cidade dos Homens)" - TV series - (Fernando Meirelles, Kátia Lund, Paulo Morelli, Cao Hamburger and others) - 2006 - "The Samba Poet (Noel - Poeta da Vila)" - (Ricardo Van Steen) - 2007 - "City of Men (Cidade dos Homens)" - (Paulo Morelli) - 2008 - "Embarque Imediato" - (Allan Fiterman) - 2018 - "O Mecanismo" - (José Padilha) - 2020 - "Omn
- evidence 2 (E 0.220 / N 0.772 / C 0.008): ro".
- evidence 3 (E 0.204 / N 0.795 / C 0.001): He has participated, as a model, in the Fashion Rio event, campaigning for Dolce & Gabbana and worked in promotional events for the NBA. He participated in the reality show "A Fazenda", from TV Record.
- most contradicting (E 0.121 / N 0.809 / C 0.070): iscient"
- article lead sentence: Jonathan Haagensen Jonathan Haagensen Cerqueira (born February 23, 1983) is a Brazilian actor, model, and singer.
- category: 
- notes: 

### FN13 (SELECTED): Haagensen began his acting career in theater.

- gold S, predicted NS, score 0.0000, word overlap 0.250, PRUNED before DeBERTa
- evidence 1 (E 0.987 / N 0.012 / C 0.001): Haagensen appeared in the film "City of God", by Fernando Meireles and Kátia Lund, as did his brother, Phellipe Haagensen, also an actor. He was discovered in the Vidigal's theatrical company "Nós do Morro".
- evidence 2 (E 0.705 / N 0.293 / C 0.002): He was discovered in the Vidigal's theatrical company "Nós do Morro".
- evidence 3 (E 0.147 / N 0.848 / C 0.005): He was discovered in the Vidigal's theatrical company "Nós do Morro". He has participated, as a model, in the Fashion Rio event, campaigning for Dolce & Gabbana and worked in promotional events for the NBA.
- most contradicting (E 0.041 / N 0.822 / C 0.137): iscient"
- article lead sentence: Jonathan Haagensen Jonathan Haagensen Cerqueira (born February 23, 1983) is a Brazilian actor, model, and singer.
- category: 
- notes: 

### FN14: Haagensen later transitioned to television.

- gold S, predicted NS, score 0.0000, word overlap 0.000, PRUNED before DeBERTa
- evidence 1 (E 0.017 / N 0.957 / C 0.026): 2000 - "Palace II" - Short - (Fernando Meirelles and Kátia Lund) - 2002 - "City of God (Cidade de Deus)" - (Fernando Meirelles and Kátia Lund) - 2002 - "Seja o que Deus Quiser" - (Murilo Salles) - 2004 - "Four for None (O Diabo a Quatro)" - (Alice de Andrade) - 2002/05 - "City of Men (Cidade dos Homens)" - TV series - (Fernando Meirelles, Kátia Lund, Paulo Morelli, Cao Hamburger and others) - 2006 - "The Samba Poet (Noel - Poeta da Vila)" - (Ricardo Van Steen) - 2007 - "City of Men (Cidade dos Homens)" - (Paulo Morelli) - 2008 - "Embarque Imediato" - (Allan Fiterman) - 2018 - "O Mecanismo" - (José Padilha) - 2020 - "Omn
- evidence 2 (E 0.014 / N 0.236 / C 0.751): He has participated, as a model, in the Fashion Rio event, campaigning for Dolce & Gabbana and worked in promotional events for the NBA. He participated in the reality show "A Fazenda", from TV Record.
- evidence 3 (E 0.013 / N 0.960 / C 0.027): The director's name is listed in parenthesis. - 2000 - "Palace II" - Short - (Fernando Meirelles and Kátia Lund) - 2002 - "City of God (Cidade de Deus)" - (Fernando Meirelles and Kátia Lund) - 2002 - "Seja o que Deus Quiser" - (Murilo Salles) - 2004 - "Four for None (O Diabo a Quatro)" - (Alice de Andrade) - 2002/05 - "City of Men (Cidade dos Homens)" - TV series - (Fernando Meirelles, Kátia Lund, Paulo Morelli, Cao Hamburger and others) - 2006 - "The Samba Poet (Noel - Poeta da Vila)" - (Ricardo Van Steen) - 2007 - "City of Men (Cidade dos Homens)" - (Paulo Morelli) - 2008 - "Embarque Imediato" - (Allan Fiterman) - 2018 - "O Mecanismo" - (José Padilha) - 2020 - "Omn
- most contradicting (E 0.006 / N 0.139 / C 0.856): He participated in the reality show "A Fazenda", from TV Record.
- article lead sentence: Jonathan Haagensen Jonathan Haagensen Cerqueira (born February 23, 1983) is a Brazilian actor, model, and singer.
- category: 
- notes: 

### FN15 (SELECTED): They divorced in 1964.

- gold S, predicted NS, score 0.0000, word overlap 0.000, PRUNED before DeBERTa
- evidence 1 (E 0.003 / N 0.157 / C 0.840): Egyptian government.
- evidence 2 (E 0.003 / N 0.858 / C 0.139): Her first marriage, at age sixteen, was to Ardeshir Zahedi on 11 October 1957, at Golestan Palace in Tehran. He was one-time Iranian foreign minister and twice Iranian ambassador to the United States (1957–64 and 1972–79).{{cite news|title=Iran Shah's daughter to wed engineer in simple ceremony|url=https://news.google.com/newspapers?id=ZcdGAAAAIBAJ&sjid=8vMMAAAAIBAJ&pg=1761,3998465&dq=marble+palace+tehran&hl=en|access-date=23 July 2013|newspaper=Lewiston Evening Journal Shahnaz later married Khosrow Jahanbani in February 1971 at the Iranian Embassy, Paris.{{cite web|title=Centers of Power in Iran During her father's reign, Shahnaz had investments in agricultural enterprises and assembly plants of Honda bicycles and motorcycles.
- evidence 3 (E 0.002 / N 0.165 / C 0.833): Honours. -
- most contradicting (E 0.000 / N 0.005 / C 0.995): He was one-time Iranian foreign minister and twice Iranian ambassador to the United States (1957–64 and 1972–79).{{cite news|title=Iran Shah's daughter to wed engineer in simple ceremony|url=https://news.google.com/newspapers?id=ZcdGAAAAIBAJ&sjid=8vMMAAAAIBAJ&pg=1761,3998465&dq=marble+palace+tehran&hl=en|access-date=23 July 2013|newspaper=Lewiston Evening Journal Shahnaz later married Khosrow Jahanbani in February 1971 at the Iranian Embassy, Paris.{{cite web|title=Centers of Power in Iran During her father's reign, Shahnaz had investments in agricultural enterprises and assembly plants of Honda bicycles and motorcycles.
- article lead sentence: Shahnaz Pahlavi Shahnaz Pahlavi (, born 27 October 1940) is the first child of the Shah of Iran, Mohammad Reza Pahlavi, and his first wife, Princess Fawzia of Egypt.
- category: 
- notes: 

### FN16 (SELECTED): He has since then starred in numerous dramas.

- gold S, predicted NS, score 0.0700, word overlap 0.500
- evidence 1 (E 0.070 / N 0.920 / C 0.010): He is best known for his roles in the television dramas "All In" (2003), "Save the Last Dance for Me" (2004), "New Heart" (2007), "Protect the Boss" (2011), "Secret Love" (2013), "Kill Me, Heal Me" (2015), "Innocent Defendant" (2017), "Familiar Wife" (2018), "Doctor John" (2019), "The Devil Judge" (2021) and the film "My PS Partner" (2012).
- evidence 2 (E 0.045 / N 0.951 / C 0.005): Ji Sung Ji Sung (born Kwak Tae-geun on 27 February 1977) is a South Korean actor. He is best known for his roles in the television dramas "All In" (2003), "Save the Last Dance for Me" (2004), "New Heart" (2007), "Protect the Boss" (2011), "Secret Love" (2013), "Kill Me, Heal Me" (2015), "Innocent Defendant" (2017), "Familiar Wife" (2018), "Doctor John" (2019), "The Devil Judge" (2021) and the film "My PS Partner" (2012).
- evidence 3 (E 0.005 / N 0.995 / C 0.000): " (2004), he played a rich amnesiac businessman who falls in love with a girl running a bed and breakfast who nurses him back to health (played by Eugene). He then made a guest appearance playing Bae Jong-ok's college boyfriend in "Outing", a two-episode arc written by Lee Kyung-hee for the omnibus drama "Beating Heart" (2005), followed by a supporting role in Kim Dae-seung's period thriller "Blood Rain" (2005).
- most contradicting (E 0.000 / N 0.923 / C 0.076): nesic prosecutor who finds himself on a death row.
- article lead sentence: Ji Sung Ji Sung (born Kwak Tae-geun on 27 February 1977) is a South Korean actor.
- category: 
- notes: 

### FN17 (SELECTED): He has since then starred in numerous films.

- gold S, predicted NS, score 0.0203, word overlap 0.500
- evidence 1 (E 0.020 / N 0.978 / C 0.002): He is best known for his roles in the television dramas "All In" (2003), "Save the Last Dance for Me" (2004), "New Heart" (2007), "Protect the Boss" (2011), "Secret Love" (2013), "Kill Me, Heal Me" (2015), "Innocent Defendant" (2017), "Familiar Wife" (2018), "Doctor John" (2019), "The Devil Judge" (2021) and the film "My PS Partner" (2012).
- evidence 2 (E 0.013 / N 0.986 / C 0.002): Ji Sung Ji Sung (born Kwak Tae-geun on 27 February 1977) is a South Korean actor. He is best known for his roles in the television dramas "All In" (2003), "Save the Last Dance for Me" (2004), "New Heart" (2007), "Protect the Boss" (2011), "Secret Love" (2013), "Kill Me, Heal Me" (2015), "Innocent Defendant" (2017), "Familiar Wife" (2018), "Doctor John" (2019), "The Devil Judge" (2021) and the film "My PS Partner" (2012).
- evidence 3 (E 0.007 / N 0.992 / C 0.002): Career.:2008–2014: Comeback and lead roles.
- most contradicting (E 0.000 / N 0.195 / C 0.804): nesic prosecutor who finds himself on a death row.
- article lead sentence: Ji Sung Ji Sung (born Kwak Tae-geun on 27 February 1977) is a South Korean actor.
- category: 
- notes: 

### FN18: "Familiar Wife." is popular.

- gold S, predicted NS, score 0.1515, word overlap 0.667
- evidence 1 (E 0.152 / N 0.846 / C 0.003): The series was a hit and developed a cult following, winning Ji Sung the "Daesang (Grand Prize)" at the MBC Drama Awards.
- evidence 2 (E 0.123 / N 0.287 / C 0.589): Early life.
- evidence 3 (E 0.120 / N 0.829 / C 0.051): Several months later, he played another historical character in "Kim Su-ro, The Iron King", as the titular founder of Geumgwan Gaya.
- most contradicting (E 0.001 / N 0.004 / C 0.995): odrama "Secret Love".
- article lead sentence: Ji Sung Ji Sung (born Kwak Tae-geun on 27 February 1977) is a South Korean actor.
- category: 
- notes: 

## Summary table to fill in

| category | definition (1-2 sentences) | FP | FN | total |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
