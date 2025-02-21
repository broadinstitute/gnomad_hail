# noqa: D100

import logging
import random
from collections import Counter
from typing import Any, Callable, List, Optional, Tuple, Union

import hail as hl
import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from gnomad.utils.filtering import filter_to_autosomes

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POP_NAMES = {
    "afr": "African/African-American",
    "ami": "Amish",
    "amr": "Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "eur": "European",
    "fin": "Finnish",
    # NOTE: mde is kept for historical purposes, in gnomAD v3.1 mid was used instead
    "mde": "Middle Eastern",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "oth": "Other",
    "remaining": "Remaining individuals",
    "sas": "South Asian",
    "uniform": "Uniform",
    "sas_non_consang": "South Asian (F < 0.05)",
    "consanguineous": "South Asian (F > 0.05)",
    "exac": "ExAC",
    "bgr": "Bulgarian (Eastern European)",
    "est": "Estonian",
    "gbr": "British",
    "nwe": "North-Western European",
    "seu": "Southern European",
    "swe": "Swedish",
    "kor": "Korean",
    "sgp": "Singaporean",
    "jpn": "Japanese",
    "oea": "Other East Asian",
    "oeu": "Other European",
    "onf": "Other Non-Finnish European",
    "unk": "Unknown",
}

POP_COLORS = {
    "afr": "#941494",
    "ami": "#FFC0CB",
    "amr": "#ED1E24",
    "asj": "#FF7F50",
    "eas": "#108C44",
    "eur": "#6AA5CD",
    "fin": "#002F6C",
    "mde": "#33CC33",
    "nfe": "#6AA5CD",
    "oth": "#ABB9B9",
    "sas": "#FF9912",
    "uniform": "pink",
    "consanguineous": "pink",
    "sas_non_consang": "orange",
    "exac": "gray",
    "bgr": "#66C2A5",
    "est": "black",
    "gbr": "#C60C30",
    "nwe": "#C60C30",
    "seu": "#3CA021",
    "swe": "purple",
    "kor": "#4891D9",
    "sgp": "darkred",
    "jpn": "#BC002D",
    "oea": "#108C44",
    "oeu": "#6AA5CD",
    "onf": "#6AA5CD",
    "unk": "#ABB9B9",
    "remaining": "#ABB9B9",
    "": "#ABB9B9",
}


def pc_project(
    mt: hl.MatrixTable,
    loadings_ht: hl.Table,
    loading_location: str = "loadings",
    af_location: str = "pca_af",
) -> hl.Table:
    """
    Project samples in `mt` on pre-computed PCs.

    :param mt: MT containing the samples to project
    :param loadings_ht: HT containing the PCA loadings and allele frequencies used for the PCA
    :param loading_location: Location of expression for loadings in `loadings_ht`
    :param af_location: Location of expression for allele frequency in `loadings_ht`
    :return: Table with scores calculated from loadings in column `scores`
    """
    n_variants = loadings_ht.count()

    mt = mt.annotate_rows(
        pca_loadings=loadings_ht[mt.row_key][loading_location],
        pca_af=loadings_ht[mt.row_key][af_location],
    )

    mt = mt.filter_rows(
        hl.is_defined(mt.pca_loadings)
        & hl.is_defined(mt.pca_af)
        & (mt.pca_af > 0)
        & (mt.pca_af < 1)
    )

    gt_norm = (mt.GT.n_alt_alleles() - 2 * mt.pca_af) / hl.sqrt(
        n_variants * 2 * mt.pca_af * (1 - mt.pca_af)
    )

    mt = mt.annotate_cols(scores=hl.agg.array_sum(mt.pca_loadings * gt_norm))

    return mt.cols().select("scores")


def apply_onnx_classification_model(
    data_pd: pd.DataFrame, fit: onnx.ModelProto
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply an ONNX classification model `fit` to a pandas dataframe `data_pd`.

    :param data_pd: Pandas dataframe containing the data to be classified.
    :param fit: ONNX model to be applied.
    :return: Tuple of classification and probabilities.
    """
    if not isinstance(fit, onnx.ModelProto):
        raise TypeError("The model supplied is not an onnx model!")

    sess = rt.InferenceSession(
        fit.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name
    classification = sess.run([label_name], {input_name: data_pd.astype(np.float32)})[0]
    probs = sess.run([prob_name], {input_name: data_pd.astype(np.float32)})[0]
    probs = pd.DataFrame.from_dict(probs)
    probs = probs.add_prefix("prob_")

    return classification, probs


def apply_sklearn_classification_model(
    data_pd: pd.DataFrame, fit: Any
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply an sklearn classification model `fit` to a pandas dataframe `data_pd`.

    :param data_pd: Pandas dataframe containing the data to be classified.
    :param fit: Sklearn model to be applied.
    :return: Tuple of classification and probabilities.
    """
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(fit)
    except TypeError:
        raise TypeError("The supplied model is not an sklearn model!")

    logger.warning(
        "sklearn models have different rounding behavior than ONNX models. This may "
        "lead to subtly different assignment results for samples around probability "
        "cutoffs."
    )
    classification = fit.predict(data_pd)
    probs = fit.predict_proba(data_pd)
    probs = pd.DataFrame(probs, columns=[f"prob_{p}" for p in fit.classes_])

    return classification, probs


def convert_sklearn_rf_to_onnx(
    fit: Any, target_opset: Optional[int] = None
) -> onnx.ModelProto:
    """
    Convert a sklearn random forest model to ONNX.

    :param fit: Sklearn random forest model to be converted.
    :param target_opset: An optional target ONNX opset version to convert the model to.
    :return: ONNX model.
    """
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(fit)
    except TypeError:
        raise TypeError("The supplied model is not an sklearn model!")

    initial_type = [("float_input", FloatTensorType([None, fit.n_features_in_]))]
    onx = convert_sklearn(fit, initial_types=initial_type, target_opset=target_opset)

    domains = onx.opset_import
    opset_version = ""
    for dom in domains:
        opset_version += f"domain: {dom.domain}, version: {dom.version}\n"

    logger.info(
        "sklearn model converted to onnx model with the following opset version: \n%s",
        opset_version,
    )

    return onx


def assign_population_pcs(
    pop_pca_scores: Union[hl.Table, pd.DataFrame],
    pc_cols: Union[hl.expr.ArrayExpression, List[int], List[str]],
    known_col: str = "known_pop",
    fit: Any = None,  # Type should be RandomForestClassifier but we do not want to import sklearn.RandomForestClassifier outside
    seed: int = 42,
    prop_train: float = 0.8,
    n_estimators: int = 100,
    min_prob: float = 0.9,
    output_col: str = "pop",
    missing_label: str = "oth",
    pc_expr: Union[hl.expr.ArrayExpression, str] = "scores",
    convert_model_func: Optional[Callable[[Any], Any]] = None,
    apply_model_func: Callable[
        [pd.DataFrame, Any], Any
    ] = apply_sklearn_classification_model,
) -> Tuple[
    Union[hl.Table, pd.DataFrame], Any
]:  # 2nd element of the tuple should be RandomForestClassifier but we do not want to import sklearn.RandomForestClassifier outside
    """
    Use a random forest model to assign population labels based on the results of PCA.

    Default values for model and assignment parameters are those used in gnomAD.

    As input, this function can either take:
        - A Hail Table (typically the output of `hwe_normalized_pca`). In this case,
            - `pc_cols` should be one of::
                - A list of integers where each element is one of the PCs to use.
                - A list of strings where each element is one of the PCs to use.
                - An ArrayExpression of Floats where each element is one of the PCs.
                  to use
            - A Hail Table will be returned as output.
        - A Pandas DataFrame. In this case:
            - Each PC should be in a separate column and `pc_cols` is the list of all
              the columns containing the PCs to use.
            - A pandas DataFrame is returned as output.

    .. note::

        If you have a Pandas Dataframe and have all PCs as an array in a single column,
        the `expand_pd_array_col`can be used to expand this column into multiple `PC`
        columns.

    :param pop_pca_scores: Input Hail Table or Pandas Dataframe.
    :param pc_cols: List of which PCs to use/columns storing the PCs to use. Values
        provided should be 1-based and should be a list of integers when passing in a
        Hail Table (i.e. [1, 2, 4, 5]) or a list of strings when passing in a Pandas
        Dataframe (i.e. ["PC1", "PC2", "PC4", "PC5"]). When passing a HT this can also
        be an ArrayExpression containing all the PCs to use.
    :param known_col: Column storing the known population labels.
    :param fit: Fit from a previously trained random forest model (i.e., the output
        from a previous RandomForestClassifier() call).
    :param seed: Random seed.
    :param prop_train: Proportion of known data used for training.
    :param n_estimators: Number of trees to use in the RF model.
    :param min_prob: Minimum probability of belonging to a given population for the
        population to be set (otherwise set to `None`).
    :param output_col: Output column storing the assigned population.
    :param missing_label: Label for samples for which the assignment probability is
        smaller than `min_prob`.
    :param pc_expr: Column storing the list of PCs. Only used if `pc_cols` is a List of
        integers. Default is scores.
    :param convert_model_func: Optional function to convert the model to ONNX format.
        Default is no conversion.
    :param apply_model_func: Function to apply the model to the data. Default is
        `apply_sklearn_classification_model`, which will apply a sklearn classification
        model to the data. This default will work if no `fit` is set, or the supplied
        `fit` is a sklearn classification model.
    :return: Hail Table or Pandas Dataframe (depending on input) containing sample IDs
        and imputed population labels, trained random forest model.
    """
    from sklearn.ensemble import RandomForestClassifier

    hail_input = isinstance(pop_pca_scores, hl.Table)
    if hail_input:
        if isinstance(pc_cols, list):
            if not all(isinstance(n, int) for n in pc_cols):
                raise TypeError(
                    "Using a Hail Table with a list of PC cols to use (pc_cols) "
                    "requires all values of the pc_cols list to be integers."
                )
            if isinstance(pc_expr, str):
                pc_expr = pop_pca_scores[pc_expr]
            pcs_to_pull = [pc_expr[i - 1] for i in pc_cols]
        else:
            pc_col_len = list(
                filter(
                    None,
                    pop_pca_scores.aggregate(hl.agg.collect_as_set(hl.len(pc_cols))),
                )
            )
            if len(pc_col_len) > 1:
                raise ValueError(
                    "More than one length was found among the 'pc_cols' "
                    "ArrayExpression values. The length must be consistent!"
                )
            pcs_to_pull = pc_cols
            pc_cols = list(range(1, pc_col_len[0] + 1))
        if not fit:
            pop_pca_scores = pop_pca_scores.select(known_col, pca_scores=pcs_to_pull)
        else:
            pop_pca_scores = pop_pca_scores.select(pca_scores=pcs_to_pull)

        pop_pc_pd = pop_pca_scores.to_pandas()

        # Explode the PC array
        pc_cols = [f"PC{i}" for i in pc_cols]
        pop_pc_pd[pc_cols] = pd.DataFrame(pop_pc_pd["pca_scores"].values.tolist())

    else:
        if not all(isinstance(n, str) for n in pc_cols):
            raise TypeError(
                "Using a Pandas DataFrame with pc_cols requires all values of the"
                " pc_cols list to be strings."
            )
        pop_pc_pd = pop_pca_scores

    # Split training data into subsamples for fitting and evaluating.
    if not fit:
        train_data = pop_pc_pd.loc[~pop_pc_pd[known_col].isnull()]
        N = len(train_data)
        random.seed(seed)
        train_subsample_ridx = random.sample(list(range(0, N)), int(N * prop_train))
        train_fit = train_data.iloc[train_subsample_ridx]
        fit_samples = [x for x in train_fit["s"]]
        evaluate_fit = train_data.loc[~train_data["s"].isin(fit_samples)]

        # Train RF.
        training_set_known_labels = train_fit[known_col].values
        training_set_pcs = train_fit[pc_cols].values
        evaluation_set_pcs = evaluate_fit[pc_cols].values

        pop_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        pop_clf.fit(training_set_pcs, training_set_known_labels)
        logger.info(
            "Random forest feature importances are as follows: %s",
            pop_clf.feature_importances_,
        )

        # Evaluate RF.
        predictions = pop_clf.predict(evaluation_set_pcs)
        error_rate = 1 - sum(evaluate_fit[known_col] == predictions) / float(
            len(predictions)
        )
        logger.info("Estimated error rate for RF model is %.4f", error_rate)
    else:
        pop_clf = fit

    # Classify data.
    classifications, probs = apply_model_func(pop_pc_pd[pc_cols].values, pop_clf)

    pop_pc_pd[output_col] = classifications
    pop_pc_pd = pd.concat(
        [pop_pc_pd.reset_index(drop=True), probs.reset_index(drop=True)], axis=1
    )
    probs["max"] = probs.max(axis=1)
    pop_pc_pd.loc[probs["max"] < min_prob, output_col] = missing_label
    pop_pc_pd = pop_pc_pd.drop(pc_cols, axis="columns")

    logger.info(
        "Found the following sample count after population assignment: %s",
        ", ".join(
            f"{pop}: {count}" for pop, count in Counter(pop_pc_pd[output_col]).items()
        ),
    )

    if convert_model_func is not None:
        pop_clf = convert_model_func(pop_clf)

    if hail_input:
        pops_ht = hl.Table.from_pandas(pop_pc_pd, key=list(pop_pca_scores.key))
        pops_ht = pops_ht.annotate_globals(
            assign_pops_from_pc_params=hl.struct(min_assignment_prob=min_prob)
        )

        if not fit:
            pops_ht = pops_ht.annotate_globals(
                assign_pops_from_pc_params=pops_ht.assign_pops_from_pc_params.annotate(
                    error_rate=error_rate
                )
            )

            pops_ht = pops_ht.annotate(
                evaluation_sample=hl.literal(list(evaluate_fit.s)).contains(pops_ht.s),
                training_sample=hl.literal(list(train_fit.s)).contains(pops_ht.s),
            )
        return pops_ht, pop_clf
    else:
        return pop_pc_pd, pop_clf


def run_pca_with_relateds(
    qc_mt: hl.MatrixTable,
    related_samples_to_drop: Optional[hl.Table] = None,
    additional_samples_to_drop: Optional[hl.Table] = None,
    n_pcs: int = 10,
    autosomes_only: bool = True,
) -> Tuple[List[float], hl.Table, hl.Table]:
    """
    Run PCA excluding the given related or additional samples, and project those samples in the PC space to return scores for all samples.

    The `related_samples_to_drop` and `additional_samples_to_drop` Tables have to be keyed by the sample ID and all samples present in these
    tables will be excluded from the PCA.

    The loadings Table returned also contains a `pca_af` annotation which is the allele frequency
    used for PCA. This is useful to project other samples in the PC space.

    :param qc_mt: Input QC MT
    :param related_samples_to_drop: Optional table of related samples to drop when generating the PCs, these samples will be projected in the PC space
    :param additional_samples_to_drop: Optional table of additional samples to drop when generating the PCs, these samples will be projected in the PC space
    :param n_pcs: Number of PCs to compute
    :param autosomes_only: Whether to run the analysis on autosomes only
    :return: eigenvalues, scores and loadings
    """
    if autosomes_only:
        qc_mt = filter_to_autosomes(qc_mt)

    # 'pca_mt' is the MatrixTable to use for generating the PCs
    # If samples to drop are provided in 'related_samples_to_drop' or
    # 'additional_samples_to_drop', 'project_pca_mt' will also be generated
    # and will contain the samples to project in the PC space
    pca_mt = qc_mt

    if related_samples_to_drop:
        pca_mt = pca_mt.filter_cols(
            hl.is_missing(related_samples_to_drop[pca_mt.col_key])
        )
    if additional_samples_to_drop:
        pca_mt = pca_mt.filter_cols(
            hl.is_missing(additional_samples_to_drop[pca_mt.col_key])
        )

    pca_evals, pca_scores, pca_loadings = hl.hwe_normalized_pca(
        pca_mt.GT, k=n_pcs, compute_loadings=True
    )
    pca_af_ht = pca_mt.annotate_rows(
        pca_af=hl.agg.mean(pca_mt.GT.n_alt_alleles()) / 2
    ).rows()
    pca_loadings = pca_loadings.annotate(
        pca_af=pca_af_ht[pca_loadings.key].pca_af
    )  # TODO: Evaluate if needed to write results at this point if relateds or not

    if not related_samples_to_drop and not additional_samples_to_drop:
        return pca_evals, pca_scores, pca_loadings
    else:
        pca_loadings = pca_loadings.persist()
        pca_scores = pca_scores.persist()
        project_pca_mt = qc_mt.filter_cols(hl.is_missing(pca_mt.cols()[qc_mt.col_key]))
        projected_scores = pc_project(project_pca_mt, pca_loadings)
        pca_scores = pca_scores.union(projected_scores)
        return pca_evals, pca_scores, pca_loadings
