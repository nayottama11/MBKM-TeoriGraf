# Teori-Graf

Repository ini berisi implementasi beberapa algoritma **Teori Graf** menggunakan bahasa pemrograman **Python**.

---

## Anggota 
- Giandra Raihan Nayottama – 5025221094
- Dimas Ahmad Fahreza – 5025221319

---

## Implementasi Tree (Minimum Spanning Tree)

### Prim’s Algorithm
Algoritma Prim digunakan untuk mencari **Minimum Spanning Tree (MST)** pada graf berbobot.

**Cara menjalankan:**
```bash
python prims.py
```

**Requirement:**
- `networkx`
- `matplotlib`

---

### Kruskal Algorithm
Algoritma Kruskal digunakan untuk mencari **Minimum Spanning Tree (MST)** dengan mengurutkan sisi berdasarkan bobot.

**Cara menjalankan:**
```bash
python kruskal.py
```

**Requirement:**
- `networkx`
- `matplotlib`

---

## Graph Traversal & Lintasan

### Chinese Postman Problem
Program ini menyelesaikan **Chinese Postman Problem**, yaitu mencari lintasan terpendek yang melewati seluruh sisi graf.

**Cara menjalankan:**
```bash
python ChinesePostmanAlgorithmRevisi.py
```

---

### Travelling Salesman Problem (TSP)
Program untuk menyelesaikan **Travelling Salesman Problem**, yaitu mencari rute terpendek yang mengunjungi setiap simpul tepat satu kali.

**Cara menjalankan:**
```bash
python travelling.py
```

---

## 🎯 Matching & Coloring

### Hungarian Algorithm
Digunakan untuk menyelesaikan **Assignment Problem** (masalah penugasan) dengan biaya minimum.

**Cara menjalankan:**
```bash
python hungarian.py
```

---

### Welsh-Powell Algorithm
Digunakan untuk melakukan **pewarnaan graf** berdasarkan derajat simpul menggunakan algoritma Welsh-Powell.

**Cara menjalankan:**
```bash
python welsh_powell.py
```

---

## Persiapan & Instalasi

Pastikan Python 3 sudah terpasang.  
Install library yang dibutuhkan dengan perintah:

```bash
pip install networkx matplotlib
```

> Catatan: Tidak semua program membutuhkan library eksternal.

---

## Struktur File
File utama dalam repository ini:
- `prims.py`
- `kruskal.py`
- `ChinesePostmanAlgorithm.py`
- `ChinesePostmanAlgorithmRevisi.py`
- `travelling.py`
- `hungarian.py`
- `welsh_powell.py`

---


