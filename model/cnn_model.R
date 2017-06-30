library(keras)

### Reading ##################################################################

# For training, keras has built in MNIST datset

test_raw <- read.csv("data/test.csv")
test.mat <- as.matrix(test_raw)/255      #normalizing test data
test.mat <- aperm(`dim<-`(t(test.mat), list(28,28,28000,1)), c(3,2,1,4)) 
# converting 2d test matrix into 4d array 

### processing data for convolution network ###################################
mnist <- dataset_mnist()

train.x <- mnist$train$x
train.y <- mnist$train$y

valid.x <- mnist$test$x
valid.y <- mnist$test$y

train_img <- array(train.x, dim = c(dim(train.x), 1))/255   # normalizing and adding one more dimension
train.label <- to_categorical(train.y, 10)                  # one-hot encoding labels

valid_img <- array(valid.x, dim = c(dim(valid.x), 1))/255
valid.label <- to_categorical(valid.y, 10)

### Building Model #####################################################################

model <- keras_model_sequential()

model %>%
      layer_conv_2d(filters = 28, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28, 28, 1), data_format = "channels_last") %>%        # dim(?, 28, 28, 28)
      layer_conv_2d(filters = 28, kernel_size = c(3,3), activation = "relu") %>%                                                                                     # dim(?, 26, 26, 28)                       
      layer_max_pooling_2d(pool_size = c(2,2)) %>%                                                                                                                   # dim(?, 13, 13, 28)            
      layer_dropout(rate = 0.2) %>%
      layer_flatten() %>%                                                                                                                                            # dim(?, 4732)                  
      layer_dense(units = 10, activation = "relu") %>%                                                                                                               # dim(?, 10) 
      layer_dense(units = 10, activation = "softmax")                                                                                                                # dim(?, 10)                


model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_nadam(lr = 0.002),     # nadam = Adam, RMSprop with Nesterov momentum
      metrics = c('accuracy')
)


history <- model %>% fit(
      train_img, train.label, 
      epochs = 10,
      batch_size = 128,
      validation_split = 0.1,
      callbacks = callback_tensorboard(log_dir = "logs/run_cnn1")    # saving logs for tensorboard visulizations
)

tensorboard()

### Saving model and weights ####
save_model_hdf5(model, "saves/save_cnn1.h5")
save_model_weights_hdf5(model, "saves/save_cnn1_weights.h5")

plot(history)

loss_and_metrics <- model %>%
      evaluate(valid_img, valid.label, batch_size= 128)


### Predicting on test set ####
classes <- model %>% predict(test.mat)

n_class <- max.col(classes)-1


### submission ##################################################################

submission <- read.csv("data/submission.csv")
submission$Label <- n_class
write.csv(submission, "keras_cnn1.csv", row.names = F)


