# -*- coding: utf-8 -*-
"""
Created on Sun May 13 15:32:39 2018

@author: Pap
"""
import csv, re, gensim
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
from operator import itemgetter
import itertools
from collections import OrderedDict

news = input()
doc = news.lower().split('.')
file = open("stopwordbahasa.csv", 'r')
reader = csv.reader(file)
stopWords =  set([x for y in [row for row in reader] for x in y] + StopWordRemoverFactory().get_stop_words())
stemmer = StemmerFactory().create_stemmer()

def preprocessing(doc):
    punc_free = re.sub(r'[^\w]', ' ', doc)
    stop_free = " ".join([x for x in punc_free.split() if x not in stopWords])
    num_free = ' '.join([x for x in stop_free.split() if any(y.isdigit() for y in x) == False])
    #stem_free = stemmer.stem(num_free)
    res = ' '.join([x for x in num_free.split() if len(x)>2])
    return res

doc_clean = [preprocessing(doc_i).split() for doc_i in doc]
doc_clean2 = []
for i in range(len(doc_clean)):
    temp = []
    for j in range(len(doc_clean[i])):
        if [doc_clean[i][j] in x for x in doc_clean].count(True) >=2:
            temp += [doc_clean[i][j]]
    if temp != []:
        doc_clean2 += [temp]

#LDA -----------------------------------------------------    
dictionary = gensim.corpora.Dictionary(doc_clean2)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean2]
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary)
LDA = ldamodel.show_topics(num_topics=1, formatted=False)[0][1]

#NMF -------------------------------------------------------
sentence = [' '.join(text) for text in doc_clean2]
vectorizer = CountVectorizer(analyzer='word', max_features=5000)
x_counts = vectorizer.fit_transform(sentence)
transformer = TfidfTransformer(smooth_idf=False)
x_tfidf = transformer.fit_transform(x_counts)
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
model = NMF(n_components=10, init='nndsvd');
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    a = []
    feat_names = vectorizer.get_feature_names()
    for i in range(10):
        words_ids = model.components_[i].argsort()[:-3 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        a += [words]
    return list(Counter([x for y in a for x in y]).items());

NMF = get_nmf_topics(model, 3)
innerjoin = list(set([x[0] for x in LDA]) & set([y[0] for y in NMF]))

#Get Topic --------------------------------------------------
teknologi = ['daftar', 'april', 'batas', 'nomor', 'nik', 'prabayar', 'lewat', 'aktif', 'operator', 'nonaktif', 'duduk', 'bijak', 'tuang', 'induk', 'registrasi', 'seluler', 'kartu', 'kominfo', 'sim', 'sesuai', 'surat', 'blokir', 'langgan', 'guna', 'undang', 'varian', 'depan', 'megapiksel', 'resolusi', 'usung', 'yuan', 'harga', 'xiaomi', 'china', 'layar', 'ram', 'redmi', 'pasar', 'juta', 'target', 'banderol', 'kamera', 'ganda', 'sepakbola', 'konsumsi', 'jakarta', 'indonesia', 'rural', 'konten', 'koneksi', 'beda', 'andal', 'anggap', 'vero', 'google', 'youtube', 'area', 'kota', 'wifi', 'daerah', 'video', 'guna', 'urban', 'tonton', 'hasil', 'daftar', 'imbas', 'nomor', 'aku', 'prabayar', 'juta', 'ririek', 'aktivitas', 'registrasi', 'turun', 'dapat', 'milik', 'kartu', 'ulang', 'telkomsel', 'dorong', 'blokir', 'rugi', 'langgan', 'guna', 'alami', 'usaha', 'youtube', 'bagi', 'ios', 'putar', 'facebook', 'nikmat', 'whatsapp', 'cakap', 'android', 'guna', 'instagram', 'fitur', 'video', 'kirim', 'sedia', 'jendela', 'samsung', 'megapiksel', 'aspect', 'ratio', 'jepret', 'plus', 'layar', 'lensa', 'bekal', 'beda', 'storage', 'baterai', 'andal', 'milik', 'galaxy', 'resolusi', 'bal', 'kamera', 'ganda', 'mah', 'ram', 'inci', 'ponsel', 'samsung', 'uluwatu', 'megapiksel', 'konsumen', 'pasar', 'produk', 'indonesia', 'ukur', 'banding', 'nilai', 'marketing', 'murah', 'smartphone', 'galaxy', 'kamera', 'harga', 'gaya', 'spesifikasi', 'mah', 'guna', 'vendor', 'banderol', 'main', 'perangkat', 'database', 'ringan', 'rilis', 'duga', 'seri', 'jalan', 'android', 'produksi', 'sistem', 'lini', 'core', 'smartphone', 'kuat', 'bas', 'galaxy', 'aju', 'harga', 'rendah', 'sesuai', 'kenal', 'spesifikasi', 'operasi', 'vendor', 'ponsel', 'kartu', 'consumer', 'turun', 'paket', 'pasar', 'operator', 'indosat', 'industri', 'jangka', 'basis', 'arpu', 'registrasi', 'joy', 'voucher', 'prabayar', 'kuartal', 'sim', 'push', 'aktif', 'jakarta', 'strategi', 'imbas', 'jual', 'ooredoo', 'persero', 'driven', 'langgan', 'apple', 'pasar', 'italia', 'analis', 'prancis', 'eropa', 'negara', 'posisi', 'kuartal', 'samsung', 'rilis', 'geser', 'tempat', 'persen', 'hasil', 'persentase', 'smartphone', 'vendor', 'xiaomi', 'pangsa', 'huawei']
olahraga = ['liga', 'manfaat', 'optajose', 'sempat', 'tanding', 'boateng', 'kalah', 'ubah', 'skor', 'levante', 'barcelona', 'valencia', 'gol', 'musim', 'emmanuel', 'com', 'twitter', 'pic', 'menit', 'hasil', 'tim', 'babak', 'may', 'spanyol', 'wasit', 'laga', 'kalah', 'klasemen', 'inggris', 'liga', 'gelar', 'scorer', 'cetak', 'hasil', 'champions', 'league', 'main', 'liverpool', 'mohamed', 'premier', 'top', 'gol', 'musim', 'salah', 'inggris', 'robin', 'liga', 'cetak', 'catat', 'van', 'main', 'liverpool', 'mohamed', 'persie', 'klopp', 'gol', 'musim', 'salah', 'liga', 'jarak', 'tiket', 'imbang', 'europa', 'inter', 'poin', 'baca', 'koleksi', 'fiorentina', 'champions', 'kompas', 'juventus', 'simy', 'italia', 'gol', 'musim', 'posisi', 'atalanta', 'cetak', 'lazio', 'com', 'langsung', 'sisa', 'hasil', 'enam', 'oleh', 'main', 'sassuolo', 'mil', 'liga', 'empat', 'hotspur', 'tiket', 'harap', 'swansea', 'tentu', 'laga', 'united', 'koleksi', 'hove', 'chelsea', 'champions', 'stoke', 'gol', 'musim', 'posisi', 'city', 'the', 'albion', 'newcastle', 'inggris', 'west', 'tottenham', 'tim', 'kasta', 'menang', 'pekan', 'liverpool', 'brighton', 'duduk', 'basket', 'mas', 'repsol', 'bola', 'olahraga', 'sito', 'hormat', 'hindar', 'celaka', 'valentino', 'risiko', 'rossi', 'com', 'sengaja', 'motor', 'sentuh', 'insiden', 'pedrosa', 'sepak', 'motogp', 'marc', 'pons', 'balap', 'persija', 'klasemen', 'prasetyo', 'penalti', 'gatra', 'unggul', 'jayapura', 'kalah', 'laga', 'united', 'poin', 'gawang', 'fabiano', 'bambang', 'daryono', 'skor', 'valentino', 'tahan', 'gol', 'jakarta', 'serang', 'peluang', 'pamungkas', 'hasil', 'menit', 'beltrame', 'madura', 'telaubun', 'suarez', 'neymar', 'umpan', 'tertawa', 'rindu', 'barcelona', 'trio', 'musim', 'takut', 'jalan', 'pulih', 'psg', 'gagal', 'brasil', 'piala', 'pindah', 'messi', 'hasil', 'main', 'paris', 'raih', 'liga', 'kroos', 'atletico', 'madrid', 'bola', 'real', 'peringkat', 'unggul', 'laga', 'kalah', 'dominasi', 'poin', 'tembak', 'gol', 'hakim', 'bek', 'celta', 'milik', 'vigo', 'salip', 'hasil', 'menit', 'ton', 'babak', 'menang', 'asensio', 'duduk', 'sasar', 'spanyol', 'sergi', 'striker', 'gomez', 'persipura', 'baya', 'main', 'laga', 'menang', 'persib', 'lawan', 'bandung', 'kartu', 'airlangga', 'mukhlis', 'tim', 'senang', 'tanding', 'bauman', 'kuning', 'ezechiel']
ekonomi = ['giat', 'dollar', 'lemah', 'riset', 'kompas', 'juni', 'rupiah', 'dampak', 'rate', 'ekonomi', 'suku', 'januari', 'buruk', 'indonesia', 'tukar', 'gerak', 'bank', 'konter', 'usaha', 'kurs', 'transaksi', 'kondisi', 'asing', 'nilai', 'posisi', 'sektor', 'level', 'beli', 'dollar', 'sore', 'spot', 'com', 'bank', 'kuat', 'jual', 'level', 'senin', 'dagang', 'kompas', 'tinggal', 'rupiah', 'jakarta', 'pimpin', 'corpora', 'terbang', 'pesawat', 'premiair', 'kalah', 'skandal', 'mantan', 'malaysia', 'indonesia', 'satrio', 'razak', 'pilih', 'mahathir', 'chairman', 'najib', 'milik', 'usaha', 'koalisi', 'peter', 'manajemen', 'jet', 'rencana', 'istri', 'rajawali', 'terang', 'pribadi', 'kontraktor', 'etatisme', 'swasta', 'kue', 'negara', 'bangun', 'infrastruktur', 'bima', 'banyak', 'bagi', 'bumn', 'jakarta', 'dpr', 'kredibilitas', 'amerika', 'kurang', 'apbn', 'fundamental', 'rupiah', 'pulih', 'belanja', 'halaman', 'zon', 'ekonomi', 'tarif', 'negara', 'kuat', 'terima', 'perintah', 'utang', 'indonesia', 'negeri', 'sebab', 'tingkat', 'percaya', 'bijak', 'kondisi', 'terbit', 'fadli', 'global', 'posisi', 'pajak', 'nilai', 'sosialisasi', 'bulan', 'online', 'kolaborasi', 'program', 'sejahtera', 'jamin', 'kota', 'bpjs', 'mati', 'jalan', 'daftar', 'lindung', 'ketenagakerjaan', 'driver', 'risiko', 'ribu', 'bayar', 'tingkat', 'serta', 'masyarakat', 'literasi', 'mudah', 'mitra', 'iur', 'jek', 'paham', 'uang', 'juta', 'jakarta', 'bentuk', 'dollar', 'valas', 'total', 'milik', 'non', 'dolar', 'tugas', 'finansial', 'utang', 'gerak', 'kondisi', 'triliun', 'persen', 'bima', 'bengkak', 'rupiah', 'bumn', 'jakarta', 'kerap', 'logistik', 'orang', 'biaya', 'bangun', 'pindah', 'infrastruktur', 'tuju', 'wajib', 'operasional', 'kurang', 'persen', 'bima', 'bumn', 'jakarta', 'bulan', 'pinjam', 'kta', 'produk', 'jamin', 'beda', 'dana', 'karyawan', 'bilang', 'agun', 'kredit', 'aju', 'bank', 'tenor', 'milik', 'lebih', 'syarat', 'proses', 'wirausaha', 'anggap', 'situs', 'banding', 'butuh', 'tahu', 'halomoney', 'nilai', 'transaksi', 'com', 'pertamina', 'lemah', 'rupiah', 'impor', 'hedging', 'nicke', 'laku', 'dolar', 'kompas']
new = ['tutup', 'brimob', 'tembak', 'mako', 'prencje', 'teroris', 'duga', 'jaring', 'polri', 'gugur', 'lelaki', 'orang', 'khusus', 'mobil', 'parkir', 'mewah', 'evaluasi', 'sumarno', 'peristiwa', 'warga', 'tangkap', 'ambil', 'kampung', 'jadi', 'bripka', 'tuju', 'polisi', 'sosok', 'tinggal', 'duga', 'cadar', 'ledak', 'armuji', 'geledah', 'dita', 'orang', 'tutur', 'ngobrol', 'istri', 'bom', 'salah', 'baca', 'warga', 'rumah', 'keluarga', 'kaget', 'usaha', 'surabaya', 'sosok', 'waspada', 'polrestabes', 'sepeda', 'serang', 'bom', 'barung', 'berita', 'mapolrestabes', 'surabaya', 'kompas', 'asal', 'frans', 'mobil', 'motor', 'wib', 'ledak', 'sidang', 'dpr', 'selesai', 'juni', 'akibat', 'sidoarjo', 'luka', 'revisi', 'orang', 'cepat', 'senin', 'kompas', 'jokowi', 'korban', 'antiterorisme', 'surabaya', 'presiden', 'bom', 'wonocolo', 'jalan', 'teror', 'tewas', 'perppu', 'gereja', 'ledak', 'medis', 'icjr', 'terorisme', 'jamin', 'proses', 'dunia', 'langsung', 'regulasi', 'putus', 'saksi', 'perintah', 'pasal', 'bantu', 'korban', 'anggara', 'negara', 'penuh', 'laku', 'ayat', 'kompensasi', 'hak', 'adil', 'tinggal', 'lindung', 'tito', 'kapolri', 'indonesia', 'polisi', 'proses', 'jad', 'kelompok', 'kait', 'orang', 'isis', 'kompas', 'antiterorisme', 'hukum', 'suriah', 'surabaya', 'aksi', 'bom', 'presiden', 'utama', 'baca', 'com', 'laku', 'keluarga', 'tangkap', 'sel', 'pulang', 'jat', 'darah', 'bhayangkara', 'wib', 'bawa', 'tekan', 'luka', 'orang', 'anak', 'sus', 'motor', 'kompas', 'korban', 'surabaya', 'jalan', 'baca', 'com', 'keluarga', 'drop', 'terobos', 'operasi', 'tewas', 'gereja', 'evan', 'wenny', 'cerita', 'ledak', 'jarang', 'puji', 'koesni', 'beli', 'suami', 'rusiono', 'banyuwangi', 'mobil', 'anak', 'tiga', 'senin', 'pakai', 'kompas', 'surabaya', 'bom', 'gki', 'baca', 'com', 'dita', 'keluarga', 'jual', 'nikah', 'tutup', 'ledak', 'magetan', 'mako', 'mapolrestabes', 'polisi', 'indrianto', 'kapolda', 'supra', 'masuk', 'arifin', 'tinjau', 'anak', 'ledak', 'motor', 'lempar', 'usia', 'kompas', 'jaga', 'surabaya', 'bom', 'identifikasi', 'com', 'laku', 'eko', 'suwarso', 'selamat', 'pasca', 'timur', 'bonceng', 'beat', 'minggu', 'rangkai', 'tni', 'polisi', 'peristiwa', 'bunuh', 'temu', 'sidoarjo', 'gantung', 'orang', 'nyawa', 'bantu', 'moeldoko', 'kompas', 'senin', 'surabaya', 'bom', 'presiden', 'baca', 'com', 'butuh', 'satu', 'polri', 'jakarta', 'staf', 'ledak']
full = list(set([stemmer.stem(x[0]) for x in LDA] + [stemmer.stem(y[0]) for y in NMF]))
allClass = [teknologi, olahraga, ekonomi, new]
topic = ["Teknologi", "Olahraga", "Ekonomi", "News"]
prob = []
for i in range(len(allClass)):
    prob += [len(list(set(full) & set(allClass[i])))]
print("Topic: ", topic[prob.index(max(prob))])

#Get Tag -----------------------------------------------------
allTopic = []
for i in range(len(innerjoin)):
    allTopic += [(innerjoin[i], [x for x in LDA if innerjoin[i] in x][0][1]*[x for x in NMF if innerjoin[i] in x][0][1])]
weight = sorted(allTopic, key=itemgetter(1), reverse=True)
topic = [x[0] for x in weight]
tag = []
for i in range(len(topic)):
    if topic[i].title() in news:
        tag += [topic[i].title()]
match = list(itertools.combinations(topic,2)) + list(itertools.combinations(topic,3))
for i in range(len(match)):
    temp = ' '.join(match[i])
    if temp in news.lower():
        tag += [temp.title()]

print("Tag:", ', '.join(tag))

#Get Title --------------------------------------------------
tmp = [re.sub(r'[^\w]', ' ', x).split() for x in news.lower().split('.')[:3]]
lst = [x for x in tmp if x != min(tmp, key=len)]
term = list(set([x[0] for x in LDA + NMF]))
prob = []
for i in range(len(lst)):
    prob += [len(set(lst[i]) & set(term)) / len(lst[i])]
title = list(OrderedDict.fromkeys([x.title() for x in lst[prob.index(max(prob))] if x in term]))
print("Title:", ' '.join(title))
