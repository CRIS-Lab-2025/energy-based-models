def epoch(s):
    for x, y in self._dataloader:

            # inference (free phase relaxation)
            self._network.set_input(x, reset=False)  # we set the input, and we let the state of the network where it was at the end of the previous batch
            self._energy_minimizer.compute_equilibrium()  # we let the network settle to equilibrium (free state)
            self._cost_fn.set_target(y)  # we present the correct (desired) output
            self._do_measurements(0)  # we measure the statistics of the free state (energy value, cost value, error value, ...)

            # training step
            grads = self._differentiator.compute_gradient()  # compute the parameter gradients
            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._do_measurements(1)  # measure the statistics of training
            self._optimizer.step()  # perform one step of gradient descent on the parameters (of both the energy function E and the cost function C)
            for param in self._params: param.clamp_()  # cl