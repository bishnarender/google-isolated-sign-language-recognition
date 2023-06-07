import tensorflow as tf



class AWP(tf.keras.Model):
    '''
        Adversarial Weight Perturbation (AWP) technique within the training step.
    '''
    def __init__(self, *args, delta=0.1, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        # delta is small perturbation.        
        self.delta = delta
        
        self.eps = eps
        self.start_step = start_step
        
    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        
        # tf.GradientTape() => Record operations for automatic differentiation.
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # resources held by a GradientTape are released as soon as GradientTape.gradient() method is called.
        # As the other answers describe, it is used to record ("tape") a sequence of operations performed upon some input and producing some output, so that the output can be differentiated with respect to the input (via backpropagation / reverse-mode autodiff) (in order to then perform gradient descent optimisation).        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        
        for i in range(len(params_gradients)):            
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            # for each trainable variable, it computes the perturbation delta by dividing the product of self.delta and the gradients by the square root of the sum of squared gradients plus self.eps. 
            # this perturbation encourages the model to move away from the current weights in a direction that minimizes the loss. 
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            # .assign_add(value) => update ref by adding value to it.
            self.trainable_variables[i].assign_add(delta)
            
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            # calculates the loss again, but this time with the perturbed weights.
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
            
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
            
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            # this time reversing the previous perturbations by subtracting delta from trainable variables.
            # .assign_sub(value) => update ref by subtracting value from it.
            self.trainable_variables[i].assign_sub(delta)
            
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        # this method "overrides the default train_step (if condition is false)" method of the parent class and introduces AWP.
        # if the condition is true, it executes the standard training step by calling "super(AWP, self).train_step(data)".
        return tf.cond(self._train_counter < self.start_step, lambda:super(AWP, self).train_step(data), lambda:self.train_step_awp(data))



class FGM(tf.keras.Model):
    '''
        Fast Gradient Sign Method (FGSM) within the training step of the model.     
    '''
    def __init__(self, *args, delta=0.2, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        # delta is small perturbation.        
        self.delta = delta
        
        self.eps = eps
        self.start_step = start_step
        
    def train_step_fgm(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        # tf.GradientTape() => Record operations for automatic differentiation.
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # resources held by a GradientTape are released as soon as GradientTape.gradient() method is called.
        # As the other answers describe, it is used to record ("tape") a sequence of operations performed upon some input and producing some output, so that the output can be differentiated with respect to the input (via backpropagation / reverse-mode autodiff) (in order to then perform gradient descent optimisation).
        with tf.GradientTape() as tape:            
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        embedding = self.trainable_variables[0]
        # gradients are computed only for the first trainable variable, which is assumed to be an embedding.        
        embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0]
        embedding_gradients = tf.zeros_like (embedding) + embedding_gradients
        delta = tf.math.divide_no_nan(self.delta * embedding_gradients , tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + self.eps)
        self.trainable_variables[0].assign_add(delta)
        
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
                
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
            
        self.trainable_variables[0].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(self._train_counter < self.start_step, lambda:super(FGM, self).train_step(data), lambda:self.train_step_fgm(data))




