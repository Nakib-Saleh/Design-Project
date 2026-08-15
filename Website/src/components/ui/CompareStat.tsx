import DataValue from "./DataValue";

interface Props {
  groundTruth: string | number;
  prediction: string | number;
  unit?: string;
}

export default function CompareStat({ groundTruth, prediction, unit }: Props) {
  return (
    <div className="flex items-center gap-4 bg-surface px-4 py-3 border border-grid-line/50 w-fit">
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-ink-muted mb-1">
          Ground Truth
        </span>
        <DataValue value={groundTruth} unit={unit} tone="primary" />
      </div>
      
      <div className="w-[1px] h-8 bg-grid-line/50" />
      
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-ink-muted mb-1">
          Prediction
        </span>
        <DataValue value={prediction} unit={unit} tone="secondary" />
      </div>
    </div>
  );
}
