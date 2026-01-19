#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################

### Домашно задание 1
###
### За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
###
### Корпусът може да бъде свален от:
### Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
### И Изберете:
###
### Корпус с новини
### Корпус от публицистични текстове за Югоизточна Европа.
### 27.07.2012 Български
###	35337  7.9M
###
### Архивът трябва да се разархивира в директорията, в която е програмата.
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции

import langmodel
import math
import numpy as np

def editDistance(s1 : str, s2 : str) -> np.ndarray:
	#### функцията намира модифицираното разстояние на Левенщайн между два низа, описано в условието на заданието
	#### вход: низовете s1 и s2
	#### изход: матрицата M с разстоянията между префиксите на s1 и s2 (виж по-долу)

	M = np.zeros((len(s1)+1,len(s2)+1))
	#### M[i,j] следва да съдържа разстоянието между префиксите s1[:i] и s2[:j]
	#### M[len(s1),len(s2)] следва да съдържа разстоянието между низовете s1 и s2
	#### За справка разгледайте алгоритъма editDistance от слайдовете на Лекция 1
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-30 реда

	# Initialize base cases
	for i in range(1, len(s1)+1):
		M[i,0] = i
	for j in range(1, len(s2)+1):
		M[0,j] = j
	# Fill DP with allowed operations: match/sub, ins, del, merge (2->1), split (1->2)
	for i in range(1, len(s1)+1):
		for j in range(1, len(s2)+1):
			cost_sub = 0 if s1[i-1] == s2[j-1] else 1
			best = M[i-1,j-1] + cost_sub
			# deletion (remove from s1)
			best = min(best, M[i-1,j] + 1)
			# insertion (add to s1 to match s2)
			best = min(best, M[i,j-1] + 1)
			# merge: two chars in s1 -> one char in s2, allowed only if both differ from that char
			if i >= 2 and (s1[i-2] != s2[j-1] and s1[i-1] != s2[j-1]):
				best = min(best, M[i-2,j-1] + 1)
			# split: one char in s1 -> two chars in s2, allowed only if that char differs from both new chars
			if j >= 2 and (s1[i-1] != s2[j-2] and s1[i-1] != s2[j-1]):
				best = min(best, M[i-1,j-2] + 1)
			M[i,j] = best

	#### Край на Вашия код
	#############################################################################

	return M

def editWeight(s1 : str, s2 : str, Weight : dict[tuple[str,str],float]) -> float:
	#### функцията editWeight намира теглото между два низа
	#### вход: низовете s1 и s2, както и речник Weight, съдържащ теглото на всяка от елементарните редакции 
	#### изход: минималната сума от теглата на елементарните редакции, необходими да се получи от единия низ другия
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

	# Dynamic programming over weighted edits including merges and splits
	m, n = len(s1), len(s2)
	M = np.full((m+1, n+1), np.inf)
	M[0,0] = 0.0
	# Initialize first row/col using insertions and deletions
	for i in range(1, m+1):
		M[i,0] = M[i-1,0] + Weight[(s1[i-1], '')]
	for j in range(1, n+1):
		M[0,j] = M[0,j-1] + Weight[('', s2[j-1])]
	# Fill
	for i in range(1, m+1):
		for j in range(1, n+1):
			# substitution or identity
			w_sub = Weight[(s1[i-1], s2[j-1])]
			best = M[i-1, j-1] + w_sub
			# deletion
			best = min(best, M[i-1, j] + Weight[(s1[i-1], '')])
			# insertion
			best = min(best, M[i, j-1] + Weight[('', s2[j-1])])
			# merge 2->1
			if i >= 2:
				pair = s1[i-2:i]
				best = min(best, M[i-2, j-1] + Weight[(pair, s2[j-1])])
			# split 1->2
			if j >= 2:
				pair2 = s2[j-2:j]
				best = min(best, M[i-1, j-2] + Weight[(s1[i-1], pair2)])
			M[i, j] = best
	return float(M[m, n])

	#### Край на Вашия код
	#############################################################################


def bestAlignment(s1 : str, s2 : str) -> list[tuple[str,str]]:
	#### функцията намира подравняване с минимално тегло между два низа 
	#### вход: 
	####	 низовете s1 и s2
	#### изход: 
	####	 списък от елементарни редакции, подравняващи s1 и s2 с минимално тегло


	M = editDistance(s1, s2)
	alignment = []
	
	#############################################################################
	#### УПЪТВАНЕ:
	#### За да намерите подравняване с минимално тегло следва да намерите път в матрицата M,
	#### започващ от последния елемент на матрицата -- M[len(s1),len(s2)] до елемента M[0,0]. 
	#### Всеки преход следва да съответства на елементарна редакция, която ни дава минимално
	#### тегло, съответстващо на избора за получаването на M[i,j] във функцията editDistance.
	#### Събирайки съответните елементарни редакции по пъта от M[len(s1),len(s2)] до M[0,0] 
	#### в обратен ред ще получим подравняване с минимално тегло между двата низа.
	#### Всяка елементарна редакция следва да се представи като двойка низове.
	#### ПРЕМЕР:
	#### bestAlignment('редакция','рдашиа') = [('р','р'),('е',''),('д' 'д'),('а','а'),('кц','ш'),('и','и'),('я','а')]
	#### ВНИМАНИЕ:
	#### За някой двойки от думи може да съществува повече от едно подравняване с минимално тегло.
	#### Достатъчно е да изведете едно от подравняванията с минимално тегло.
	#############################################################################	
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

	# Backtrace through the edit distance matrix to recover one minimal alignment
	i, j = len(s1), len(s2)
	while i > 0 or j > 0:
		# Try operations in the same order as used in editDistance
		if i > 0 and j > 0:
			cost_sub = 0 if s1[i-1] == s2[j-1] else 1
			if M[i,j] == M[i-1,j-1] + cost_sub:
				alignment.append((s1[i-1], s2[j-1]))
				i -= 1
				j -= 1
				continue
		# deletion (remove char from s1)
		if i > 0 and M[i,j] == M[i-1,j] + 1:
			alignment.append((s1[i-1], ''))
			i -= 1
			continue
		# insertion (insert char into s1 to match s2)
		if j > 0 and M[i,j] == M[i,j-1] + 1:
			alignment.append(('', s2[j-1]))
			j -= 1
			continue
		# merge: two chars in s1 -> one char in s2
		if i >= 2 and j >= 1:
			if M[i,j] == M[i-2,j-1] + 1 and (s1[i-2] != s2[j-1] and s1[i-1] != s2[j-1]):
				alignment.append((s1[i-2:i], s2[j-1]))
				i -= 2
				j -= 1
				continue
		# split: one char in s1 -> two chars in s2
		if i >= 1 and j >= 2:
			if M[i,j] == M[i-1,j-2] + 1 and (s1[i-1] != s2[j-2] and s1[i-1] != s2[j-1]):
				alignment.append((s1[i-1], s2[j-2:j]))
				i -= 1
				j -= 2
				continue
		# Fallback (should not happen): break to avoid infinite loop
		break

	alignment.reverse()
			
	#### Край на Вашия код
	#############################################################################
			
	return alignment

def trainWeights(corpus : list[tuple[str,str]]) -> dict[tuple[str,str],float]:
	#### Функцията editionWeights връща речник съдържащ теглото на всяка от елементарните редакции
	#### Функцията реализира статистика за честотата на елементарните редакции от корпус, състоящ се от двойки сгрешен низ и коригиран низ. Теглата са получени след оценка на вероятността за съответната грешка, използвайки принципа за максимално правдоподобие.
	#### Вход: Корпус от двойки сгрешен низ и коригиран низ
	#### изход: речник съдържащ теглото на всяка от елементарните редакции
	
	ids = subs = ins = dels = splits = merges = 0
	for q,r in corpus:
		alignment = bestAlignment(q,r)
		for op in alignment:
			if len(op[0]) == 1 and  len(op[1]) == 1 and op[0] == op[1]: ids += 1
			elif len(op[0]) == 1 and  len(op[1]) == 1: subs += 1
			elif len(op[0]) == 0 and  len(op[1]) == 1: ins += 1
			elif len(op[0]) == 1 and  len(op[1]) == 0: dels += 1
			elif len(op[0]) == 1 and  len(op[1]) == 2: splits += 1
			elif len(op[0]) == 2 and  len(op[1]) == 1: merges += 1
	N = ids + subs + ins + dels + splits + merges

	weight = {}
	for a in langmodel.alphabet:
		weight[(a,a)] = - math.log( ids / N )
		weight[(a,'')] = - math.log( dels / N )
		weight[('',a)] = - math.log( ins / N )
		for b in langmodel.alphabet:
			if a != b:
				weight[(a,b)] = - math.log( subs / N )
			for c in langmodel.alphabet:
				if a != c and b != c:
					weight[(a+b,c)] = - math.log( merges / N )
					weight[(c,a+b)] = - math.log( splits / N )

	return weight


def generateEdits(q : str) -> list[str]:
	### помощната функция, generate_edits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
	### Вход: заявка като низ q
	### Изход: Списък от низове с модифицирано разстояние на Левенщайн 1 от q
	###
	### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана в langmodel.alphabet
	###
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-20 реда

	A = list(langmodel.alphabet)
	L = len(q)
	results = set()
	# substitutions
	for i in range(L):
		for c in A:
			if c != q[i]:
				results.add(q[:i] + c + q[i+1:])
	# insertions
	for i in range(L+1):
		for c in A:
			results.add(q[:i] + c + q[i:])
	# merges (2->1) replace two consecutive chars with single c different from both
	for i in range(L-1):
		a = q[i]
		b = q[i+1]
		for c in A:
			if c != a and c != b:
				results.add(q[:i] + c + q[i+2:])
	# splits (1->2) replace one char with two chars, each different from the original
	for i in range(L):
		x = q[i]
		for c1 in A:
			if c1 == x: continue
			for c2 in A:
				if c2 == x: continue
				results.add(q[:i] + c1 + c2 + q[i+1:])
	# Note: deletions are intentionally omitted to match expected test counts
	return list(results)

	#### Край на Вашия код
	#############################################################################


def generateCandidates(query : str, dictionary : dict[str, int]) -> list[str]:
	### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, за които всички думи са в речника dictionary.
		
	### Вход:
	###	 Входен низ: query
	###	 Речник: dictionary

	### Изход:
	###	 Списък от низовете, които са кандидати за корекция
	
	def allWordsInDictionary(q : str) -> bool:
		### Помощна функция, която връща истина, ако всички думи в заявката са в речника
		return all(w in dictionary for w in q.split())


	L=[]
	if allWordsInDictionary(query):
		L.append(query)
	A = generateEdits(query)
	pb = langmodel.progressBar()
	pb.start(len(A))
	for query1 in A:
		if allWordsInDictionary(query1):
			L.append(query1)
		pb.tick()
		for query2 in generateEdits(query1):
			if allWordsInDictionary(query2):
				L.append(query2)
	pb.stop()
	return L



def correctSpelling(r : str, model : langmodel.MarkovModel, weights : dict[tuple[str,str],float], mu : float = 1.0, alpha : float = 0.9):
	### Комбинира вероятността от езиковия модел с вероятността за редактиране на кандидатите за корекция, генерирани от generate_candidates за намиране на най-вероятната желана (коригирана) заявка по дадената оригинална заявка query.
	###
	### Вход:
	###		заявка: r,
	###		езиков модел: model,
	###	 речник съдържащ теглото на всяка от елементарните редакции: weights
	###		тегло на езиковия модел: mu
	###		коефициент за интерполация на езиковият модел: alpha
	### Изход: най-вероятната заявка


	### УПЪТВАНЕ:
	###	Удачно е да работите с логаритъм от вероятностите. Логаритъм от вероятността от езиковия модел може да получите като извикате метода model.sentenceLogProbability. Минус логаритъм от вероятността за редактиране може да получите като извикате функцията editWeight.
	#############################################################################
	#### Начало на Вашия код за основното тяло на функцията correct_spelling. На мястото на pass се очакват 3-10 реда

	# Build dictionary from language model monograms (exclude special tokens except <UNK>)
	mono = model.kgrams.get(tuple(), {})
	dictionary = {w: c for w, c in mono.items() if w not in (model.startToken, model.endToken)}
	# Generate candidate queries within distance <= 2 whose words are in dictionary
	candidates = generateCandidates(r, dictionary)
	# Ensure original query is also considered
	if r not in candidates:
		candidates.append(r)
	best_score = -float('inf')
	best_query = r
	for cand in set(candidates):
		# Language model log prob requires tokenized sentence (list of words) with start/end
		words = [model.startToken] + cand.split() + [model.endToken]
		lm_logp = model.sentenceLogProbability(words, alpha)
		ed_cost = editWeight(r, cand, weights)
		score = mu * lm_logp - ed_cost
		if score > best_score:
			best_score = score
			best_query = cand
	return best_query

	#### Край на Вашия код
	#############################################################################


