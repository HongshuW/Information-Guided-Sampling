(set-logic SLIA)

(synth-fun f ((_arg_0 String)) String
    ((Start String) (ntString String) (ntInt Int) (ntBool Bool))
    ((Start String (ntString))
    (ntString String (_arg_0 "" " " "US" "CAN" (str.++ ntString ntString) (str.replace ntString ntString ntString) (str.at ntString ntInt) (str.from_int ntInt) (ite ntBool ntString ntString) (str.substr ntString ntInt ntInt)))
    (ntInt Int (1 0 (- 1) (+ ntInt ntInt) (- ntInt ntInt) (str.len ntString) (str.to_int ntString) (ite ntBool ntInt ntInt) (str.indexof ntString ntString ntInt)))
    (ntBool Bool (true false (= ntInt ntInt) (str.prefixof ntString ntString) (str.suffixof ntString ntString) (str.contains ntString ntString)))))

(constraint (= (f "Mining US") "Mining"))
(constraint (= (f "Soybean Farming CAN") "Soybean Farming"))
(constraint (= (f "Soybean Farming") "Soybean Farming"))
(constraint (= (f "Oil Extraction US") "Oil Extraction"))
(constraint (= (f "Fishing") "Fishing"))

(check-synth)
