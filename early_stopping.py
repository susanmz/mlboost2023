import numpy as np
from ngboost import NGBRegressor
from ngboost.scores import LogScore
from ngboost.ngboost import NGBoost
from ngboost.distns import MultivariateNormal, Normal, k_categorical
from ngboost.learners import default_tree_learner
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array, check_random_state, check_X_y


class earlyNGBRegressor(NGBRegressor):
    """
    Constructor for NGBoost regression models with early stopping.

    earlyNGBRegressor is a wrapper for the NGBRegressor class that facilitates 
    early stopping with RMSE/NLL loss. The model will stop training when the 
    validation loss stops decreasing for the specified number of iterations.
    """

    def __init__(
            self,
            Dist=Normal,
            Score=LogScore,
            Base=default_tree_learner,
            natural_gradient=True,
            n_estimators=500,
            learning_rate=0.01,
            minibatch_frac=1.0,
            col_sample=1.0,
            verbose=True,
            verbose_eval=100,
            tol=1e-4,
            random_state=None,
            validation_fraction=0.1,
            early_stopping_rounds=None,
        ):
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
        )
    
    def set_early_stopping(self, type = "NLL", early_stopping_rounds = None):

        self.early_stopping_rounds = early_stopping_rounds
        self.type = type

    def fit(self, X, Y, X_val=None, Y_val=None, sample_weight=None, val_sample_weight=None, train_loss_monitor=None, val_loss_monitor=None):
        
        return_val = None

        if self.early_stopping_rounds is None:
            return_val = super().fit(X, Y, X_val, Y_val, sample_weight, val_sample_weight, train_loss_monitor, val_loss_monitor)
        else:
            if X_val is None or Y_val is None:
                raise ValueError("Early stopping requires a validation set")
            if self.type == "NLL" or self.type == "RMSE":
                self.base_models = []
                self.scalings = []
                self.col_idxs = []
                return_val = self.partial_fit(X, Y, X_val, Y_val, sample_weight, val_sample_weight, train_loss_monitor, val_loss_monitor)
            else:
                raise ValueError("Early stopping type must be 'NLL' or 'RMSE'")
                
        return return_val

    def partial_fit(self, X, Y, X_val=None, Y_val=None, sample_weight=None, val_sample_weight=None, train_loss_monitor=None, val_loss_monitor=None, early_stopping_rounds=None):
        if self.early_stopping_rounds is None:
            return super().partial_fit(X, Y, X_val, Y_val, sample_weight, val_sample_weight, train_loss_monitor, val_loss_monitor)

        if len(self.base_models) != len(self.scalings) or len(self.base_models) != len(self.col_idxs):
            raise RuntimeError(
                "Base models, scalings, and col_idxs are not the same length"
            )

        # if early stopping is specified, split X,Y and sample weights (if given) into training and validation sets
        # This will overwrite any X_val and Y_val values passed by the user directly.
        if self.early_stopping_rounds is not None:

            early_stopping_rounds = self.early_stopping_rounds

        if Y is None:
            raise ValueError("y cannot be None")

        X, Y = check_X_y(
            X, Y, accept_sparse=True, y_numeric=True, multi_output=self.multi_output
        )

        self.n_features = X.shape[1]

        loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)

        if X_val is not None and Y_val is not None:
            X_val, Y_val = check_X_y(
                X_val,
                Y_val,
                accept_sparse=True,
                y_numeric=True,
                multi_output=self.multi_output,
            )
            val_params = self.pred_param(X_val)
            val_loss_list = []
            best_val_loss = np.inf

        if not train_loss_monitor: 
            train_loss_monitor = lambda D, Y, W: D.total_score(Y, sample_weight=W)

        if not val_loss_monitor:
            if self.type == "NLL":
                #val_loss_monitor = lambda D, Y: -D.logpdf(Y).mean()
                val_loss_monitor = lambda D, Y: D.total_score(  # NOQA
                    Y, sample_weight=val_sample_weight
                )  # NOQA
            elif self.type == "RMSE":
                val_loss_monitor = lambda D, Y: mean_squared_error(D.mean(), Y, squared=False)

        for itr in range(len(self.col_idxs), self.n_estimators + len(self.col_idxs)):
            _, col_idx, X_batch, Y_batch, weight_batch, P_batch = self.sample(
                X, Y, sample_weight, params
            )
            self.col_idxs.append(col_idx)

            D = self.Manifold(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch, weight_batch)]
            loss = loss_list[-1]
            grads = D.grad(Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, weight_batch)
            scale = self.line_search(proj_grad, P_batch, Y_batch, weight_batch)

            # pdb.set_trace()
            params -= (
                self.learning_rate
                * scale
                * np.array([m.predict(X[:, col_idx]) for m in self.base_models[-1]]).T
            )

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= (
                    self.learning_rate
                    * scale
                    * np.array(
                        [m.predict(X_val[:, col_idx]) for m in self.base_models[-1]]
                    ).T
                )
                val_loss = val_loss_monitor(self.Manifold(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if val_loss < best_val_loss:
                    best_val_loss, self.best_val_loss_itr = val_loss, itr
                if (
                    early_stopping_rounds is not None
                    and len(val_loss_list) > early_stopping_rounds
                    and best_val_loss
                    < np.min(np.array(val_loss_list[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                        print(
                            f"== Best iteration / VAL{self.best_val_loss_itr} (val_loss={best_val_loss:.4f})"
                        )
                    break

            if (
                self.verbose
                and int(self.verbose_eval) > 0
                and itr % int(self.verbose_eval) == 0
            ):
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                    f"norm={grad_norm:.4f}"
                )

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result["train"] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result["val"] = {metric: val_loss_list}

        return self

