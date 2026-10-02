export type Severity = "info" | "warning" | "high" | "critical";

export const SEVERITIES: Severity[] = ["critical", "high", "warning", "info"];

export const CATEGORY_TITLES: Record<string, string> = {
  schema: "Schema & structure",
  missingness: "Missingness",
  duplicates: "Duplicates",
  types: "Types & rules",
  numeric: "Numeric",
  categorical: "Categorical",
  strings: "Text hygiene",
  temporal: "Temporal",
  relationships: "Relationships",
  distribution: "Distribution",
  target: "Target",
};

export interface Finding {
  id: string;
  detector: string;
  code: string;
  category: string;
  severity: Severity;
  title: string;
  observation: string;
  columns: string[];
  evidence: Record<string, unknown>;
  interpretation: string | null;
  suggestion: string | null;
  confidence: number | null;
  rule: string | null;
}

export interface DatasetInfo {
  name: string;
  source: string | null;
  format: string | null;
  rows: number;
  columns: number;
  memory_bytes: number;
  sampling: { method: string; rows_read: number; rows_total: number | null } | null;
  analysis_sample_rows: number | null;
}

export interface SchemaField {
  name: string;
  position: number;
  dtype: string;
  type: string;
  nullable: boolean;
  missing: number;
  missing_ratio: number;
  unique: number;
  unique_ratio: number;
  memory_bytes: number;
  hints: string[];
  reasons: string[];
}

export interface Point {
  t: string;
  v: number | null;
}

export interface NumericProfile {
  count: number;
  mean?: number;
  std?: number;
  min?: number;
  q1?: number;
  median?: number;
  q3?: number;
  max?: number;
  iqr?: number;
  skewness?: number | null;
  excess_kurtosis?: number | null;
}

export interface ColumnProfile {
  numeric?: NumericProfile;
  histogram?: { edges: number[]; counts: number[] };
  categorical?: {
    distinct: number;
    entropy_bits: number;
    normalized_entropy: number;
    top: { label: string; count: number; share: number }[];
    rare_levels: number;
  };
}

export interface DetectorDoc {
  name: string;
  category: string;
  description: string;
  assumptions: string;
  limitations: string;
}

export interface Report {
  kind: "inspection" | "comparison";
  datasi_version: string;
  created_at: string;
  dataset: DatasetInfo;
  reference: DatasetInfo | null;
  time_column: string | null;
  target: string | null;
  schema: { fields: SchemaField[]; type_counts: Record<string, number> };
  findings: Finding[];
  profiles: Record<string, ColumnProfile>;
  sections: {
    missingness?: { columns: Record<string, { count: number; ratio: number }>; missing_cells: number; total_cells: number };
    temporal?: { time_column: string; period: string; volume: Point[] };
    correlation?: { numeric_columns?: string[]; pearson?: (number | null)[][]; spearman?: (number | null)[][] };
    target?: { column: string; task: string; classes?: { label: string; share: number }[]; feature_scores?: { column: string; score: number | null }[] };
    drift?: { columns: Record<string, unknown>[]; domain_classifier_auc: number | null };
  };
  detectors: { name: string; status: string; seconds: number; findings: number; error: string | null }[];
  summary: {
    findings: number;
    by_severity: Record<Severity, number>;
    by_category: Record<string, number>;
    columns_affected: number;
    headline: string[];
  };
  recommendations: { severity: Severity; suggestion: string; because: string; finding_id: string; columns: string[] }[];
  detector_docs?: Record<string, DetectorDoc>;
}
