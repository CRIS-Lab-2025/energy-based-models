def training_epoch(s):
    for x, y in self._dataloader:

            # inference (free phase relaxation)
            W,S = self._network.set_input(x, reset=False)  # we set the input, and we let the state of the network where it was at the end of the previous batch
            W,S = self._energy_minimizer.compute_equilibrium(W,S,B)  # we let the network settle to equilibrium (free state)        

            # training step
            grads = self._differentiator.compute_gradient(W, S, B, target)  # compute the parameter gradients
            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._optimizer.step()  # perform one step of gradient descent on the parameters (of both the energy function E and the cost function C)
            for param in self._params: param.clamp_()  # cl