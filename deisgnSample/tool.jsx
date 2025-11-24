import React, { useEffect, useMemo, useRef, useState } from "react";

// ---------- Page constants (A4 @ 150DPI) ----------
const PAGE_W = 1575;
const PAGE_H = 2316;

// Default styles
const COLORS = {
    ink: "#000",
    light: "#999",
    bg: "#fff",
    select: "#3b82f6",
};

// Snap settings
const SNAP = 1; // 1px precise; change to 2,4,8 for stronger snap

// ---------- Helpers ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const snap = (v, s = SNAP) => Math.round(v / s) * s;

// Deterministic 32-bit hash for a number -> pseudo bits
function hash32(n) {
    let x = Math.imul(n ^ 0x9e3779b1, 0x85ebca6b);
    x ^= x >>> 13;
    x = Math.imul(x, 0xc2b2ae35);
    x ^= x >>> 16;
    return x >>> 0;
}

/**
 * generateAruco4x4Pattern(id:number) -> 4x4 boolean matrix
 * Placeholder visual generator (NOT the official DICT_4X4_50 sequence).
 */
function generateAruco4x4Pattern(id) {
    const bits = [];
    let h = hash32(id);
    for (let i = 0; i < 16; i++) {
        h = hash32(h + i * 2654435761);
        bits.push((h & 1) === 1 ? 1 : 0);
    }
    bits[0] ^= id & 1;
    bits[15] ^= (id >>> 1) & 1;
    const m = new Array(4).fill(0).map(() => new Array(4).fill(0));
    bits.forEach((b, i) => (m[(i / 4) | 0][i % 4] = b));
    return m;
}

// Make choice labels by spec like "A,B,C,..." or "1,2,3" or custom CSV
function makeLabels(numChoices, labelsText = "A,B,C,...") {
    const t = (labelsText || "").trim();
    if (!t) return Array.from({ length: numChoices }, (_, i) => String.fromCharCode(65 + i));
    if (t.includes(",")) {
        const arr = t.split(",").map((s) => s.trim()).filter(Boolean);
        while (arr.length < numChoices) arr.push(String.fromCharCode(65 + arr.length));
        return arr.slice(0, numChoices);
    }
    if (t.toLowerCase().startsWith("1")) {
        return Array.from({ length: numChoices }, (_, i) => String(i + 1));
    }
    return Array.from({ length: numChoices }, (_, i) => String.fromCharCode(65 + i));
}

// ---------- Element models ----------
const ElemType = {
    ARUCO: "aruco",
    ANSWER_SHEET: "answerSheet",
    ID_PANEL: "idPanel",
    TEXT_LINES: "textLines",
};

function newAruco(id = 10, x = 120, y = 120, size = 140) {
    return { id: crypto.randomUUID(), type: ElemType.ARUCO, arucoId: id, x, y, size };
}

// Updated AnswerSheet model with sheetId, labels, bubbleMargin, r
function newAnswerSheet(x, y, startNum, numQuestions, numChoices, extras = {}) {
    return {
        id: crypto.randomUUID(),
        type: ElemType.ANSWER_SHEET,
        x,
        y,
        sheetId: extras.sheetId ?? "sheet-1",
        title: "Answer Sheet",
        startNum,
        numQuestions,
        numChoices,
        labelsText: extras.labelsText ?? "A,B,C,...",
        panelW: 420,
        panelH: 480,
        topGap: 18,
        sideMargin: 28,
        bottomMargin: 24,
        bubbleMargin: extras.bubbleMargin ?? 6,
        r: extras.r ?? 13,
    };
}

function newIdPanel(x, y) {
    return {
        id: crypto.randomUUID(),
        type: ElemType.ID_PANEL,
        x,
        y,
        title: "Student ID",
        cols: 6,
        rows: 10,
        panelW: 420,
        panelH: 480,
        topBoxH: 36,
        topGap: 18,
        sideMargin: 28,
        bottomMargin: 24,
        r: 13,
    };
}
function newTextLines(x, y) {
    return {
        id: crypto.randomUUID(),
        type: ElemType.TEXT_LINES,
        x,
        y,
        items: ["Name", "Quiz", "Class", "Score"],
        line: 48,
        lineLen: 340,
        leftGap: 120,
        font: 16,
    };
}

// ---------- Rulers ----------
function Rulers() {
    const step = 50;
    const marksX = new Array(Math.ceil(PAGE_W / step)).fill(0).map((_, i) => i * step);
    const marksY = new Array(Math.ceil(PAGE_H / step)).fill(0).map((_, i) => i * step);
    return (
        <>
            <svg className="absolute top-0 left-0" width={PAGE_W} height={20}>
                {marksX.map((x) => (
                    <g key={x}>
                        <line x1={x} y1={0} x2={x} y2={20} stroke={COLORS.light} strokeWidth={1} />
                        <text x={x + 2} y={14} fontSize={10} fill={COLORS.light}>
                            {x}
                        </text>
                    </g>
                ))}
            </svg>
            <svg className="absolute top-0 left-0" width={20} height={PAGE_H}>
                {marksY.map((y) => (
                    <g key={y}>
                        <line x1={0} y1={y} x2={20} y2={y} stroke={COLORS.light} strokeWidth={1} />
                        <text x={2} y={y + 10} fontSize={10} fill={COLORS.light}>
                            {y}
                        </text>
                    </g>
                ))}
            </svg>
        </>
    );
}

// ---------- Small UI controls ----------
function NumberField({ label, value, onChange }) {
    return (
        <label className="flex items-center justify-between gap-2 mb-2">
            <span className="text-gray-600">{label}</span>
            <input
                className="w-28 border rounded px-2 py-1"
                type="number"
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
            />
        </label>
    );
}
function TextField({ label, value, onChange }) {
    return (
        <label className="flex items-center justify-between gap-2 mb-2">
            <span className="text-gray-600">{label}</span>
            <input
                className="w-40 border rounded px-2 py-1"
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
            />
        </label>
    );
}
function RangeField({ label, value, min = 0, max = 30, step = 1, onChange }) {
    return (
        <label className="block mb-3">
            <div className="flex justify-between text-sm text-gray-600">
                <span>{label}</span>
                <span>{value}</span>
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="w-full"
            />
        </label>
    );
}

// ---------- AnswerSheet Input Modal (full like screenshot) ----------
function AnswerSheetModal({ onClose, onSubmit }) {
    const [sheetId, setSheetId] = useState("hi");
    const [numQuestions, setNumQuestions] = useState(25);
    const [numChoices, setNumChoices] = useState(5);
    const [startNum, setStartNum] = useState(""); // empty = 1
    const [labelsText, setLabelsText] = useState("A,B,C,...");
    const [bubbleMargin, setBubbleMargin] = useState(6);
    const [r, setR] = useState(12);

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded shadow-lg w-96">
                <h3 className="font-bold mb-4">Answer Sheet</h3>

                <TextField label="Id" value={sheetId} onChange={setSheetId} />
                <NumberField label="Number of questions" value={numQuestions} onChange={setNumQuestions} />
                <NumberField label="Number of choices" value={numChoices} onChange={setNumChoices} />
                <TextField label="Start numbering with" value={startNum} onChange={setStartNum} />
                <TextField label="Bubble labels" value={labelsText} onChange={setLabelsText} />
                <RangeField label="Bubble margins" value={bubbleMargin} min={0} max={24} step={1} onChange={setBubbleMargin} />
                <RangeField label="Bubble size" value={r} min={6} max={24} step={1} onChange={setR} />

                <div className="mt-4 flex justify-end gap-2">
                    <button className="px-4 py-2 bg-gray-300 rounded" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        className="px-4 py-2 bg-blue-600 text-white rounded"
                        onClick={() => {
                            const s = parseInt(startNum, 10);
                            onSubmit(
                                Number.isFinite(s) ? Math.max(1, s) : 1,
                                Math.max(1, Math.floor(numQuestions)),
                                Math.max(1, Math.floor(numChoices)),
                                { sheetId, labelsText, bubbleMargin, r }
                            );
                        }}
                    >
                        Add
                    </button>
                </div>
            </div>
        </div>
    );
}

// ---------- Main App ----------
export default function App() {
    const [elems, setElems] = useState(() => [
        newTextLines(120, 180),
        newIdPanel((PAGE_W - (420 * 3 + 30 * 2)) / 2, 250),
        { ...newIdPanel((PAGE_W - (420 * 3 + 30 * 2)) / 2 + 420 + 30, 250), title: "Quiz ID" },
        { ...newIdPanel((PAGE_W - (420 * 3 + 30 * 2)) / 2 + (420 + 30) * 2, 250), title: "Class ID" },
        newAruco(10, 90, 90, 140),
        newAruco(11, PAGE_W - 90 - 140, 90, 140),
        newAruco(12, PAGE_W - 90 - 140, PAGE_H - 90 - 140, 140),
        newAruco(13, 90, PAGE_H - 90 - 140, 140),
        newAruco(24, PAGE_W * 0.33, PAGE_H * 0.36, 80),
        newAruco(25, PAGE_W * 0.5, PAGE_H * 0.36, 80),
        newAruco(26, PAGE_W * 0.67, PAGE_H * 0.36, 80),
    ]);

    const [selectedId, setSelectedId] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [showAnswerSheetModal, setShowAnswerSheetModal] = useState(false);
    const svgRef = useRef(null);

    // Dragging
    const dragRef = useRef({ id: null, dx: 0, dy: 0 });

    function onPointerDown(e, el) {
        e.stopPropagation();
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const ctm = svg.getScreenCTM();
        const loc = pt.matrixTransform(ctm.inverse());
        dragRef.current = { id: el.id, dx: loc.x - el.x, dy: loc.y - el.y };
        setSelectedId(el.id);
    }
    function onPointerMove(e) {
        if (!dragRef.current.id) return;
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const ctm = svg.getScreenCTM();
        const loc = pt.matrixTransform(ctm.inverse());
        setElems((prev) =>
            prev.map((el) =>
                el.id === dragRef.current.id
                    ? { ...el, x: snap(loc.x - dragRef.current.dx), y: snap(loc.y - dragRef.current.dy) }
                    : el
            )
        );
    }
    function onPointerUp() {
        dragRef.current.id = null;
    }

    // Keyboard nudge
    useEffect(() => {
        function onKey(e) {
            if (!selectedId) return;
            const step = e.shiftKey ? 10 : 1;
            if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
                e.preventDefault();
                setElems((prev) =>
                    prev.map((el) => {
                        if (el.id !== selectedId) return el;
                        const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
                        const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
                        return { ...el, x: snap(el.x + dx), y: snap(el.y + dy) };
                    })
                );
            }
        }
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [selectedId]);

    // Inspector bindings
    const selected = useMemo(() => elems.find((e) => e.id === selectedId) || null, [elems, selectedId]);
    function updateSelected(patch) {
        if (!selected) return;
        setElems((prev) => prev.map((e) => (e.id === selected.id ? { ...e, ...patch } : e)));
    }

    // Export JSON
    function exportJSON() {
        const payload = { page: [PAGE_W, PAGE_H], elements: elems };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "layout.json";
        a.click();
    }
    function importJSON(file) {
        const r = new FileReader();
        r.onload = () => {
            try {
                const obj = JSON.parse(r.result);
                if (obj.elements) setElems(obj.elements);
            } catch {}
        };
        r.readAsText(file);
    }

    // Export PNG: render SVG to canvas
    function exportPNG() {
        const svgNode = svgRef.current;
        const serializer = new XMLSerializer();
        const svgStr = serializer.serializeToString(svgNode);
        const img = new Image();
        const blob = new Blob([svgStr], { type: "image/svg+xml" });
        const url = URL.createObjectURL(blob);
        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = PAGE_W;
            canvas.height = PAGE_H;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#fff";
            ctx.fillRect(0, 0, PAGE_W, PAGE_H);
            ctx.drawImage(img, 0, 0);
            URL.revokeObjectURL(url);
            canvas.toBlob((png) => {
                const a = document.createElement("a");
                a.href = URL.createObjectURL(png);
                a.download = "template.png";
                a.click();
            }, "image/png");
        };
        img.src = url;
    }

    // Toolbar actions
    function add(type, params = {}) {
        if (type === ElemType.ARUCO) setElems((p) => [...p, newAruco(30, 200, 200, 120)]);
        if (type === ElemType.ANSWER_SHEET) {
            const { startNum = 1, numQuestions = 10, numChoices = 5, extras = {} } = params;
            setElems((p) => [...p, newAnswerSheet(520, 1000, startNum, numQuestions, numChoices, extras)]);
        }
        if (type === ElemType.ID_PANEL) setElems((p) => [...p, newIdPanel(200, 350)]);
        if (type === ElemType.TEXT_LINES) setElems((p) => [...p, newTextLines(120, 180)]);
    }

    return (
        <div className="w-full h-screen bg-gray-100 text-gray-900">
            {/* Toolbar */}
            <div className="p-2 flex items-center gap-2 bg-white shadow sticky top-0 z-10">
                <button className="px-3 py-1 rounded bg-black text-white" onClick={() => add(ElemType.ARUCO)}>
                    + ArUco
                </button>
                <button className="px-3 py-1 rounded bg-black text-white" onClick={() => setShowAnswerSheetModal(true)}>
                    + Answer Sheet
                </button>
                <button className="px-3 py-1 rounded bg-black text-white" onClick={() => add(ElemType.ID_PANEL)}>
                    + ID Panel
                </button>
                <button className="px-3 py-1 rounded bg-black text-white" onClick={() => add(ElemType.TEXT_LINES)}>
                    + Text Lines
                </button>
                <div className="mx-4" />
                <button className="px-3 py-1 rounded bg-emerald-600 text-white" onClick={exportPNG}>
                    Export PNG
                </button>
                <button className="px-3 py-1 rounded bg-emerald-600 text-white" onClick={exportJSON}>
                    Export JSON
                </button>
                <label className="px-3 py-1 rounded bg-slate-700 text-white cursor-pointer">
                    Import JSON
                    <input
                        type="file"
                        className="hidden"
                        accept="application/json"
                        onChange={(e) => e.target.files?.[0] && importJSON(e.target.files[0])}
                    />
                </label>
                <div className="ml-auto flex items-center gap-2">
                    <span>Zoom</span>
                    <input type="range" min={0.5} max={2} step={0.1} value={zoom} onChange={(e) => setZoom(parseFloat(e.target.value))} />
                </div>
            </div>

            <div className="flex h-[calc(100vh-48px)]">
                {/* Canvas area */}
                <div className="flex-1 overflow-auto relative">
                    <div className="p-6">
                        <div className="relative inline-block" style={{ transform: `scale(${zoom})`, transformOrigin: "0 0" }}>
                            <svg
                                ref={svgRef}
                                width={PAGE_W}
                                height={PAGE_H}
                                onPointerMove={onPointerMove}
                                onPointerUp={onPointerUp}
                                onPointerLeave={onPointerUp}
                                style={{ background: COLORS.bg, boxShadow: "0 0 8px rgba(0,0,0,.15)" }}
                            >
                                {/* Rulers */}
                                <Rulers />
                                {/* Page border */}
                                <rect x={0} y={0} width={PAGE_W} height={PAGE_H} fill="#fff" stroke="#e5e7eb" />

                                {/* Elements */}
                                {elems.map((el) => (
                                    <Element key={el.id} el={el} selected={el.id === selectedId} onPointerDown={(e) => onPointerDown(e, el)} />
                                ))}
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Inspector */}
                <div className="w-80 border-l bg-white p-3 overflow-auto">
                    <h2 className="font-semibold mb-2">Inspector</h2>
                    {selected ? (
                        <div className="space-y-2 text-sm">
                            <div className="text-xs text-gray-500">Type: {selected.type}</div>
                            <NumberField label="x" value={selected.x} onChange={(v) => updateSelected({ x: snap(v) })} />
                            <NumberField label="y" value={selected.y} onChange={(v) => updateSelected({ y: snap(v) })} />

                            {selected.type === ElemType.ARUCO && (
                                <>
                                    <NumberField label="size" value={selected.size} onChange={(v) => updateSelected({ size: Math.max(20, v) })} />
                                    <NumberField label="id" value={selected.arucoId} onChange={(v) => updateSelected({ arucoId: Math.max(0, Math.floor(v)) })} />
                                </>
                            )}

                            {selected.type === ElemType.ANSWER_SHEET && (
                                <>
                                    <TextField label="id" value={selected.sheetId} onChange={(v) => updateSelected({ sheetId: v })} />
                                    <TextField label="title" value={selected.title} onChange={(v) => updateSelected({ title: v })} />
                                    <NumberField label="start number" value={selected.startNum} onChange={(v) => updateSelected({ startNum: Math.max(1, Math.floor(v)) })} />
                                    <NumberField label="#questions" value={selected.numQuestions} onChange={(v) => updateSelected({ numQuestions: Math.max(1, Math.floor(v)) })} />
                                    <NumberField label="#choices" value={selected.numChoices} onChange={(v) => updateSelected({ numChoices: Math.max(1, Math.floor(v)) })} />
                                    <TextField label="labels (e.g. A,B,C,...)" value={selected.labelsText} onChange={(v) => updateSelected({ labelsText: v })} />
                                    <NumberField label="panel width" value={selected.panelW} onChange={(v) => updateSelected({ panelW: Math.max(120, v) })} />
                                    <NumberField label="panel height" value={selected.panelH} onChange={(v) => updateSelected({ panelH: Math.max(120, v) })} />
                                    <NumberField label="gap below top" value={selected.topGap} onChange={(v) => updateSelected({ topGap: Math.max(0, v) })} />
                                    <NumberField label="side margin" value={selected.sideMargin} onChange={(v) => updateSelected({ sideMargin: Math.max(0, v) })} />
                                    <NumberField label="bottom margin" value={selected.bottomMargin} onChange={(v) => updateSelected({ bottomMargin: Math.max(0, v) })} />
                                    <NumberField label="bubble margin (px)" value={selected.bubbleMargin} onChange={(v) => updateSelected({ bubbleMargin: Math.max(0, v) })} />
                                    <NumberField label="bubble radius" value={selected.r} onChange={(v) => updateSelected({ r: Math.max(4, v) })} />
                                </>
                            )}

                            {selected.type === ElemType.ID_PANEL && (
                                <>
                                    <TextField label="title" value={selected.title} onChange={(v) => updateSelected({ title: v })} />
                                    <NumberField label="panel width" value={selected.panelW} onChange={(v) => updateSelected({ panelW: Math.max(120, v) })} />
                                    <NumberField label="panel height" value={selected.panelH} onChange={(v) => updateSelected({ panelH: Math.max(120, v) })} />
                                    <NumberField label="#cols (top boxes & digits)" value={selected.cols} onChange={(v) => updateSelected({ cols: Math.max(1, Math.floor(v)) })} />
                                    <NumberField label="#rows (0..9)" value={selected.rows} onChange={(v) => updateSelected({ rows: Math.max(1, Math.floor(v)) })} />
                                    <NumberField label="top box height" value={selected.topBoxH} onChange={(v) => updateSelected({ topBoxH: Math.max(16, v) })} />
                                    <NumberField label="gap below boxes" value={selected.topGap} onChange={(v) => updateSelected({ topGap: Math.max(0, v) })} />
                                    <NumberField label="side margin" value={selected.sideMargin} onChange={(v) => updateSelected({ sideMargin: Math.max(0, v) })} />
                                    <NumberField label="bottom margin" value={selected.bottomMargin} onChange={(v) => updateSelected({ bottomMargin: Math.max(0, v) })} />
                                    <NumberField label="bubble radius" value={selected.r} onChange={(v) => updateSelected({ r: Math.max(4, v) })} />
                                </>
                            )}

                            {selected.type === ElemType.TEXT_LINES && (
                                <>
                                    <NumberField label="line height" value={selected.line} onChange={(v) => updateSelected({ line: Math.max(12, v) })} />
                                    <NumberField label="line length" value={selected.lineLen} onChange={(v) => updateSelected({ lineLen: Math.max(40, v) })} />
                                    <NumberField label="label gap" value={selected.leftGap} onChange={(v) => updateSelected({ leftGap: Math.max(0, v) })} />
                                    <NumberField label="font px" value={selected.font} onChange={(v) => updateSelected({ font: Math.max(8, v) })} />
                                </>
                            )}

                            <button
                                className="mt-2 px-3 py-1 rounded bg-red-600 text-white"
                                onClick={() => setElems((prev) => prev.filter((e) => e.id !== selected.id))}
                            >
                                Delete
                            </button>
                        </div>
                    ) : (
                        <div className="text-sm text-gray-500">
                            Chọn một phần tử để chỉnh tọa độ & tham số. Kéo thả để di chuyển, dùng phím mũi tên để nudge (Shift ×10).
                        </div>
                    )}
                </div>
            </div>

            {showAnswerSheetModal && (
                <AnswerSheetModal
                    onClose={() => setShowAnswerSheetModal(false)}
                    onSubmit={(startNum, numQuestions, numChoices, extras) => {
                        add(ElemType.ANSWER_SHEET, { startNum, numQuestions, numChoices, extras });
                        setShowAnswerSheetModal(false);
                    }}
                />
            )}
        </div>
    );
}

// ---------- Element renderer ----------
function Element({ el, selected, onPointerDown }) {
    if (el.type === ElemType.ARUCO) return <Aruco el={el} selected={selected} onPointerDown={onPointerDown} />;
    if (el.type === ElemType.ANSWER_SHEET) return <AnswerSheet el={el} selected={selected} onPointerDown={onPointerDown} />;
    if (el.type === ElemType.ID_PANEL) return <IdPanel el={el} selected={selected} onPointerDown={onPointerDown} />;
    if (el.type === ElemType.TEXT_LINES) return <TextLines el={el} selected={selected} onPointerDown={onPointerDown} />;
    return null;
}

function Handle({ x, y, selected, onPointerDown }) {
    return (
        <g onPointerDown={onPointerDown} style={{ cursor: "move" }}>
            {selected && <rect x={x - 4} y={y - 4} width={8} height={8} fill={COLORS.select} />}
        </g>
    );
}

function Aruco({ el, selected, onPointerDown }) {
    const { x, y, size, arucoId } = el;
    const pad = Math.round(size * 0.12); // black border around 4x4 bits
    const cell = Math.floor((size - pad * 2) / 4);
    const mat = generateAruco4x4Pattern(arucoId);
    return (
        <g>
            <rect x={x} y={y} width={size} height={size} fill="#000" onPointerDown={(e) => onPointerDown(e)} />
            <rect x={x + pad} y={y + pad} width={cell * 4} height={cell * 4} fill="#fff" />
            {mat.map((row, r) =>
                row.map((b, c) => (
                    <rect key={`${r}-${c}`} x={x + pad + c * cell} y={y + pad + r * cell} width={cell} height={cell} fill={b ? "#000" : "#fff"} />
                ))
            )}
            <text x={x + 4} y={y - 4} fontSize={10} fill="#000">
                ID:{arucoId}
            </text>
            <Handle x={x} y={y} selected={selected} onPointerDown={onPointerDown} />
            {selected && <rect x={x} y={y} width={size} height={size} fill="none" stroke={COLORS.select} strokeWidth={1.5} />}
        </g>
    );
}

function AnswerSheet({ el, selected, onPointerDown }) {
    const {
        x,
        y,
        title,
        sheetId,
        startNum,
        numQuestions,
        numChoices,
        labelsText,
        panelW,
        panelH,
        topGap,
        sideMargin,
        bottomMargin,
        bubbleMargin,
        r,
    } = el;

    const gridX0 = x + sideMargin;
    const topLabelY = y + 24; // labels
    const gridY0 = y + 30 + topGap; // below labels + gap
    const gridW = panelW - sideMargin * 2;
    const gridH = panelH - (gridY0 - y) - bottomMargin;

    const colW = gridW / numChoices;
    const rowH = gridH / numQuestions;

    const labels = makeLabels(numChoices, labelsText);

    const cx = (c) => gridX0 + c * colW + bubbleMargin + (colW - 2 * bubbleMargin) / 2;
    const cy = (ridx) => gridY0 + ridx * rowH + bubbleMargin + (rowH - 2 * bubbleMargin) / 2;

    return (
        <g onPointerDown={(e) => onPointerDown(e)}>
            <rect x={x} y={y} width={panelW} height={panelH} fill="#fff" stroke="#000" strokeWidth={2} />
            <text x={x + 10} y={y - 8} fontSize={14} fill="#000">
                {title} {sheetId ? `(${sheetId})` : ""}
            </text>

            {/* top choice labels */}
            {labels.map((lb, c) => (
                <text key={c} x={gridX0 + (c + 0.5) * colW} y={topLabelY} fontSize={12} fill="#000" textAnchor="middle">
                    {lb}
                </text>
            ))}

            {/* grid */}
            {Array.from({ length: numQuestions }).map((_, rIdx) => (
                <g key={rIdx}>
                    <text x={x + sideMargin - 8} y={cy(rIdx) + 4} fontSize={11} fill="#000" textAnchor="end">
                        {startNum + rIdx}
                    </text>
                    {Array.from({ length: numChoices }).map((_, cIdx) => (
                        <circle key={cIdx} cx={cx(cIdx)} cy={cy(rIdx)} r={r} fill="#fff" stroke="#000" strokeWidth={2} />
                    ))}
                </g>
            ))}

            <Handle x={x} y={y} selected={selected} onPointerDown={onPointerDown} />
            {selected && <rect x={x} y={y} width={panelW} height={panelH} fill="none" stroke={COLORS.select} strokeWidth={1.5} />}
        </g>
    );
}

function IdPanel({ el, selected, onPointerDown }) {
    const { x, y, title, cols, rows, panelW, panelH, topBoxH, topGap, sideMargin, bottomMargin, r } = el;
    const boxW = Math.floor((panelW - 20) / cols);
    const gridX0 = x + sideMargin;
    const gridY0 = y + 10 + topBoxH + topGap;
    const gridW = panelW - sideMargin * 2;
    const gridH = panelH - (gridY0 - y) - bottomMargin;
    const colW = gridW / cols;
    const rowH = gridH / rows;

    return (
        <g onPointerDown={(e) => onPointerDown(e)}>
            <rect x={x} y={y} width={panelW} height={panelH} fill="#fff" stroke="#000" strokeWidth={2} />
            <text x={x + 10} y={y - 8} fontSize={14} fill="#000">
                {title}
            </text>
            {/* top boxes */}
            {Array.from({ length: cols }).map((_, i) => (
                <rect key={i} x={x + 10 + i * boxW} y={y + 10} width={boxW - 6} height={topBoxH} fill="#fff" stroke="#000" strokeWidth={2} />
            ))}
            {/* grid */}
            {Array.from({ length: rows }).map((_, rIdx) => (
                <g key={rIdx}>
                    <text x={x + 6} y={gridY0 + (rIdx + 0.5) * rowH + 4} fontSize={11}>
                        {rIdx}
                    </text>
                    {Array.from({ length: cols }).map((_, cIdx) => (
                        <circle key={cIdx} cx={gridX0 + (cIdx + 0.5) * colW} cy={gridY0 + (rIdx + 0.5) * rowH} r={r} fill="#fff" stroke="#000" strokeWidth={2} />
                    ))}
                </g>
            ))}
            <Handle x={x} y={y} selected={selected} onPointerDown={onPointerDown} />
            {selected && <rect x={x} y={y} width={panelW} height={panelH} fill="none" stroke={COLORS.select} strokeWidth={1.5} />}
        </g>
    );
}

function TextLines({ el, selected, onPointerDown }) {
    const { x, y, items, line, lineLen, leftGap, font } = el;
    return (
        <g onPointerDown={(e) => onPointerDown(e)}>
            {items.map((t, i) => (
                <g key={i}>
                    <text x={x} y={y + i * line} fontSize={font}>
                        {t}
                    </text>
                    <line x1={x + leftGap} y1={y + i * line - 6} x2={x + leftGap + lineLen} y2={y + i * line - 6} stroke="#000" strokeWidth={2} />
                </g>
            ))}
            <Handle x={x} y={y} selected={selected} onPointerDown={onPointerDown} />
            {selected && <rect x={x - 8} y={y - font} width={leftGap + lineLen + 12} height={items.length * line + 8} fill="none" stroke={COLORS.select} strokeWidth={1.5} />}
        </g>
    );
}
