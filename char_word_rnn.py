

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
embedded_sequences = embedding_layer(comment_input)

x = Bidirectional(CuDNNGRU(128, return_sequences=True,go_backwards=True))(embedded_sequences)
x = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences)
x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer="he_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)

char_input = Input(shape=(MAX_CHAR_SEQUENCE_LENGTH,))
embedding_layer = Embedding(nb_chars,
                            256)
char_emb_sequences = embedding_layer(char_input)
convs=[]
for i in range(1,8):
    conv = Conv1D(64,kernel_size=i,padding='valid')(char_emb_sequences)
    conv = PReLU()(conv)
    conv = Dropout(0.1)(conv)
    conv = GlobalMaxPooling1D()(conv)
    convs.append(conv)
char_merged=concatenate(convs)
merged = concatenate([avg_pool, max_pool,char_merged])
preds = Dense(6, activation='sigmoid')(merged)