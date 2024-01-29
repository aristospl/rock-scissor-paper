# rock-scissor-paper
Machine learning agent for the game rock-paper-scissors

--------------------------------------------------------

Αρχικά εισάγουμε το σύνολο των εικόνων του dataset στην μορφή:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/6991258f-e470-4325-93b1-d05106b5fc6d)

Shape of images_rock: (726, 200, 300, 3)

Shape of images_scissors: (750, 200, 300, 3)

Shape of images_paper: (712, 200, 300, 3)

Στην συνέχεια ενώνοντας τα παραπάνω arrays και δημιουργώντας τα labels έχουμε:

all_images shape: (2188, 200, 300, 3)

all_labels shape: (2188,)

και κάνοντας split σε train και test set(80%-20%) αποκτούμε τα παρακάτω:

Train set shape: (1750, 200, 300, 3) , train labels shape: (1750,)

Test set shape: (438, 200, 300, 3) , test labels shape: (438,)

Έπειτα εκτελώντας τον αλγόριθμο για το συνολικό dataset όπως το έχουμε περιγράψει παραπάνω και κάνοντας χρήση μόνο normalization και συγκεκριμένα MinMaxScaler(), διότι έχουμε εικόνες με τιμές για κάθε pixel στο εύρος [0, 255], το οποίο θέλουμε να το μετατρέψουμε στο εύρος [0, 1], χωρίς να κάνουμε μετατροπή σε grayscale και χωρίς να μειώσουμε την διάσταση των εικόνων παίρνουμε τα εξής αποτελέσματα για Ν = 1000 γύρους του παιχνιδιού:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/b13b34ea-efa7-4fe1-a833-92ded0aff2d1)

Total Gain for Logistic Regression: 494

Total Training Time for Logistic Regression: 183.2268 seconds

Total Gain for Random Forest: 578

Total Training Time for Random Forest: 34.2929 seconds

Total Gain for SVM: 629

Total Training Time for SVM: 347.8879 seconds

Total Gain for KNN: 574

Total Training Time for KNN: 0.2330 seconds

Total Gain for Neural Network: -19

Total Training Time for Neural Network: 837.6649 seconds

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/1bd025eb-f23e-4d12-9c1c-fdbdbd6cb2ea)

Total Gain for Logistic Regression: 379

Total Training Time for Logistic Regression: 183.2960 seconds

Total Gain for Random Forest: 9

Total Training Time for Random Forest: 34.0542 seconds

Total Gain for SVM: 7

Total Training Time for SVM: 350.4209 seconds

Total Gain for KNN: 29

Total Training Time for KNN: 0.2264 seconds

Total Gain for Neural Network: 243

Total Training Time for Neural Network: 330.9554 seconds

ενώ κάνοντας χρήση μόνο των 100 πρώτων εικόνων από κάθε δυνατή κίνηση(πέτρα, ψαλίδι, χαρτί) και για Ν = 100 γύρους καταλήγουμε στα εξής αποτελέσματα:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/abe801dd-513e-4b00-ae48-c31a556829dc)
![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/fcac87d6-3cbf-450e-b8b1-65338750505f)

Αυτό που παρατηρείται είναι ότι ο πράκτορας που κάνει χρήση «Logistic Regression» είναι και αυτός που πάντα καταφέρνει να έχει μία καλή επίδοση, αλλά όχι την βέλτιστη, με βάση το κατά πόσο καταφέρνει να μαζέψει χρήματα ποντάροντας επαναληπτικά. Οπότε με βάση αυτόν θα πάμε να εξετάσουμε το πρόβλημα μας.

Δεν επιλέγεται να υπολογιστεί κάποια μετρική όπως η accuracy_score, γιατί στην συγκεκριμένη περίπτωση μας ενδιαφέρει να αξιολογήσουμε το μοντέλο με βάση το πόσο καταφέρνει να κερδίζει ο πράκτορας μας σύμφωνα με τους κανόνες του παιχνιδιού, δηλαδή να παίρνει θετική βαθμολογία(ευρώ) σε περίπτωση που κερδίζει, αρνητική βαθμολογία(ευρώ) σε περίπτωση που χάνει και να μην παίρνει τίποτα σε περίπτωση ισοπαλλίας.

Στην συνέχεια εκτελέστηκε ο αλγόριθμος, χωρίς την χρήση του «normalization» και τα αποτελέσματα που πήραμε ήταν τα παρακάτω:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/ee683273-e288-443a-9890-360ca13cbc50)

Total Gain for Logistic Regression: 495

Total Training Time for Logistic Regression: 235.8142 seconds

Όπως φαίνεται ο «agent» μας καταφέρνει να πετύχει μεγαλύτερα κέρδη, πράγμα το οποίο δεν θα το περιμέναμε. Το παραπάνω κατά πάσα πιθανότητα οφείλεται και στην τυχαιόητα της επιλογής των εικόνων από το σύνολο ελέγχου.

Στην συνέχεια πρέπει να αναφερθεί ότι κατά την εκτέλεση του αλγορίθμου εκτυπωνόταν για κάθε μοντέλο το άθροισμα των εμφανίσεων τις κάθε κίνησης(πέτρα, ψαλίδι, χαρτί) και τα αντίστοιχα αθροίσματα των απαντήσεων του πράκτορα μας. Για παράδειγμα στην συγκεκριμένη περίπτωση στους 1000 γύρους παιχνιδιού, τους 330 είχαμε τυχαία επιλογή 'πέτρας', τους  373 είχαμε 'ψαλίδι' και τους υπόλοιπους 297 είχαμε 'χαρτί'. Άντίστοιχα φαίνονται και οι κινήσεις του «agent» μας όπου στην 'πέτρα' απαντάει τις περισσότερες φορές με χαρτί(165), στο ψαλίδι απαντάει τις περισσότερες φορές με 'πέτρα'(330), ενώ στο 'χαρτί' απαντάει τις περισσότερες φορές με ψαλίδι(158).

For Logistic Regression:

  True label 0: 330 times

  Agent move 0: 141 times
  
  Agent move 1: 24 times
  
  Agent move 2: 165 times

  True label 1: 373 times

  Agent move 0: 330 times
  
  Agent move 1: 31 times
  
  Agent move 2: 12 times
  
  True label 2: 297 times

  Agent move 0: 122 times
  
  Agent move 1: 158 times
  
  Agent move 2: 17 times

Στην συνέχεια χωρίς την χρήση «normalization», αφού δεν μας έδωσε καλύτερα αποτελέσματα, εφαρμόστηκε μετατροπή των εικόνων σε «grayscale» για να δούμε πώς το μοντέλο μας θα επηρεαστεί, τα αποτελέσματα φαίνονται και η μορφή των εικόνων παρακάτω:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/1da581f6-1319-496d-ac04-e44d3f0fc0f5)

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/66346d02-561c-4c07-8bc0-109db6f4820f)

Total Gain for Logistic Regression: 148

Total Training Time for Logistic Regression: 85.1988 seconds

Όπως φαίνεται παραπάνω η βαθμολογία(χρήματα που κέρδισε) ο πράκτορας μας έπεσε αρκετά, όπως άλλωστε και περιμέναμε, καθώς χάσαμε πληροφορία, μειώνοντας στην ουσία τα χαρακτηριστικά που είχαμε αρχικά κατά 3 φορές. Παρόλα αυτά όπως ήταν αναμενόμενο ο χρόνος εκπαίδευσης υποπολλαπλασιάστηκε. 

Επιπλέον, παρακάτω φαίνονται οι απντήσεις του πράκτορα μας, αυτός δυσκολεύεται να απαντήσιε σωστά, ειδικά στις περιπτώσεις όπου του εμφανίζεται 'πέτρα' ή 'ψαλίδι'.

For Logistic Regression:

  True label 0: 352 times
  
  Agent move 0: 65 times
  
  Agent move 1: 145 times
  
  Agent move 2: 142 times
  
  True label 1: 348 times

  Agent move 0: 124 times
  
  Agent move 1: 135 times
  
  Agent move 2: 89 times

  True label 2: 300 times

  Agent move 0: 41 times
  
  Agent move 1: 157 times
  
  Agent move 2: 102 times

Στην συνέχεια παρουσιάζονται αποτελέσματα κάνοντας χρήση μόνο μείωση διάστασης των εικόνων, όπου από 200*300 μετατρέπονται σε 20*30, τα ποτελέσματα και η μορφή των εικόνων φαίνονται παρακάτω:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/ea6235b2-77b7-46b1-9384-321de647249f)

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/aa056c4d-7a98-4390-b581-30c267d9141c)

Total Gain for Logistic Regression: 483

Total Training Time for Logistic Regression: 3.6771 seconds

Σύμφωνα με τα παραπάνω αποτελέσματα, παρατηρείται ότι με την μείωση διάστασης των εικόνων πετυχαίνεται πάρα πολυ μεγάλυ μείωση του χρόνου εκπαίδευσης, παρόλο που δεν χάνουμε καθόλου σε ακρίβεια του αποτελέσματος μας!

Ακόμη όπως φαίνεται παρακάτω ο «agent» μας απαντάει καλά, ειδικά κατα την περίπτωση όπου δέχεται 'ψαλίδι', πιο συγκεκριμένα από τις 351 φορές που του δόθηκε 'ψαλίδι', τις '309' απάνησε με 'πέτρα'.

For Logistic Regression:

  True label 0: 305 times

  Agent move 0: 118 times
  
  Agent move 1: 15 times
  
  Agent move 2: 172 times

  True label 1: 351 times

  Agent move 0: 309 times
  
  Agent move 1: 23 times
  
  Agent move 2: 19 times

  True label 2: 344 times

  Agent move 0: 142 times
  
  Agent move 1: 178 times
  
  Agent move 2: 24 times

Τέλος για την προεπεξεργασία των δεδομένων έχει ενδιαφέρον και η συνδιαστηκή χρήση των παραπάνω μεθόδων, παρακάτω παρουσιάζονται τα αποτελέσματα για χρηση «normalization» και μείωση διάστασης ταυτόχρονα:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/0fdcd3aa-7533-480e-aec1-e45985308ef0)

Total Gain for Logistic Regression: 487
Total Training Time for Logistic Regression: 2.1670 seconds

Σύμφωνα με τα παραπάνω ο χρόνος έχει μειωθεί κι άλλο, ενώ η βαθμολογία(χρήματα που κέρδισε) έχει αυξηθεί ελάχιστα.

Και πάλι ο πράκτορας μας καταφέρνει να απαντήσει καλύτερα στην περίπτωση που δέχεται 'ψαλίδι'. 

For Logistic Regression:

  True label 0: 339 times

  Agent move 0: 145 times
  
  Agent move 1: 14 times
  
  Agent move 2: 180 times

  True label 1: 327 times

  Agent move 0: 309 times
  
  Agent move 1: 14 times
  
  Agent move 2: 4 times

  True label 2: 334 times

  Agent move 0: 154 times
  
  Agent move 1: 170 times
  
  Agent move 2: 10 times


Να σημειωθεί ότι παρόμοια ανάλυση αποτελεσμέτων μπορεί να γίνει και από την εκτέλεση των υπόλοιπων μοντέλων('Random Forest', 'SVM', 'KNN', 'Neural Network') που παρουσιάστηκαν στην αρχή. Παρόλα αυτά αυξάνεται πολύ ο όγκος της συγκεκριμένης αναφοράς.
=

Κλείνοντας την συγκεκριμένη μελέτη, τραβήχτηκαν μερικές φωτογραφίες και χρησιμοποιήθηκαν για τον έλεγχο του κάθε μοντέλου, τα αποτελέσματα για χρήση μόνο προεπεξεργασίας των δεδομέωνων με «normalization» και η μορφή των φωτογραφιών φαίνονται παρακάτω:

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/ab5dc306-5780-422b-a63b-b5abd10f575f)

![εικόνα](https://github.com/aristospl/rock-scissor-paper/assets/157967279/bc445ce0-a8f6-4ae1-a786-edd68b7f207a)

Total Gain for Logistic Regression: 2

Από ότι φαίνεται το μοντέλο για αυτές τις διαφορετικές εικόνες, για Ν = 10 γύρους του παιχνιδιού δεν τα πάει και τόσο καλά.

Μάλιστα όπως φαίνεται και παρακάτω, μόνο κατά την εμφάνιση 'πέτρας' και 'χαρτί' καταφέρνει να απαντήσει με επιτυχία, ενώ κατά την εμφάνιση εικόνας που απεικονίζει 'ψαλίδι' δεν καταφέρνει να απαντήσει ποτέ ορθά με 'πέτρα'.

For Logistic Regression:

  True label 0: 2 times

  Agent move 0: 0 times
  
  Agent move 1: 0 times
  
  Agent move 2: 2 times

  True label 1: 6 times

  Agent move 0: 0 times
  
  Agent move 1: 4 times
  
  Agent move 2: 2 times

  True label 2: 2 times

  Agent move 0: 0 times
  
  Agent move 1: 2 times
  
  Agent move 2: 0 times




