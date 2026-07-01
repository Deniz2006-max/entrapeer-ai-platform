"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  BrainCircuit,
  ChevronRight,
  Clock,
  GitBranch,
  Loader2,
  Plus,
  Send,
  Sparkles,
  Target,
  TrendingDown,
  User,
} from "lucide-react";

// ─── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface RootCause {
  main_cause: string;
  sub_causes: string[];
}

interface ProblemTree {
  problem_type?: string;
  main_problem: string;
  industry: string;
  root_causes: RootCause[] | string[]; // support both old string[] and new RootCause[]
  confidence_score: number;
}

interface Analysis {
  thread_id: string;
  created_at: string;
  structured_problem: ProblemTree;
}

// ─── Mock seed data ─────────────────────────────────────────────────────────────

const MOCK_HISTORY: Analysis[] = [
  {
    thread_id: "74b02b84-76ed-4a63-9fb4-e5cc353a5bf0",
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    structured_problem: {
      problem_type: "Operational",
      main_problem: "Fabrika üretim hızı düşüşü",
      industry: "İmalat",
      root_causes: [
        { main_cause: "Vardiya yönetimi eksikliği", sub_causes: ["3 vardiyaya geçiş planlanmadı", "Süpervizör açığı oluştu"] },
        { main_cause: "Motivasyon kaybı", sub_causes: ["Anket mekanizması yok", "Performans geri bildirimi eksik"] },
      ],
      confidence_score: 0.95,
    },
  },
  {
    thread_id: "a3c1f2e4-89bb-4d12-9123-cc7711aa8899",
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    structured_problem: {
      problem_type: "Growth",
      main_problem: "Satış ekibi hedef tutturamıyor",
      industry: "Satış & Pazarlama",
      root_causes: [
        { main_cause: "CRM eksikliği", sub_causes: ["Pipeline görünürlüğü yok", "Lead takibi manuel"] },
        { main_cause: "Eğitim yetersizliği", sub_causes: ["Ürün bilgisi güncel değil", "Koçluk mekanizması kurulmamış"] },
      ],
      confidence_score: 0.88,
    },
  },
];

// ─── Helpers ────────────────────────────────────────────────────────────────────

function timeAgo(isoDate: string): string {
  const diff = (Date.now() - new Date(isoDate).getTime()) / 1000;
  if (diff < 60) return "Az önce";
  if (diff < 3600) return `${Math.floor(diff / 60)} dk önce`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} sa önce`;
  return `${Math.floor(diff / 86400)} gün önce`;
}

function scoreColor(score: number) {
  if (score >= 0.9) return "text-emerald-400";
  if (score >= 0.7) return "text-amber-400";
  return "text-rose-400";
}

// ─── Sub-components ─────────────────────────────────────────────────────────────

function Sidebar({
  history,
  activeThread,
  onSelectThread,
  onNewChat,
}: {
  history: Analysis[];
  activeThread: string | null;
  onSelectThread: (a: Analysis) => void;
  onNewChat: () => void;
}) {
  return (
    <aside className="w-72 flex flex-col border-r border-slate-800 bg-slate-950 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
        <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-indigo-600 shadow-lg shadow-indigo-900/50">
          <BrainCircuit className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold text-slate-100 leading-tight">Business Agent Pro</p>
          <p className="text-xs text-slate-500">by ENTRAPEER</p>
        </div>
      </div>

      {/* New analysis button */}
      <div className="px-4 pt-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg
                     bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700
                     text-white text-sm font-medium transition-colors shadow-md shadow-indigo-900/40"
        >
          <Plus className="w-4 h-4" />
          Yeni Analiz Başlat
        </button>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto px-4 pt-5 pb-4 space-y-1">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-1">
          Geçmiş Analizler
        </p>
        {history.length === 0 && (
          <p className="text-xs text-slate-600 px-1">Henüz analiz yok.</p>
        )}
        {history.map((item) => (
          <button
            key={item.thread_id}
            onClick={() => onSelectThread(item)}
            className={`w-full text-left group rounded-lg px-3 py-2.5 transition-colors
              ${
                activeThread === item.thread_id
                  ? "bg-slate-800 text-slate-100"
                  : "hover:bg-slate-900 text-slate-400 hover:text-slate-200"
              }`}
          >
            <div className="flex items-start gap-2">
              <GitBranch className="w-3.5 h-3.5 mt-0.5 shrink-0 text-indigo-500" />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium truncate leading-snug">
                  {item.structured_problem.main_problem}
                </p>
                <div className="flex items-center gap-1.5 mt-1">
                  <Clock className="w-3 h-3 text-slate-600" />
                  <span className="text-[10px] text-slate-600">{timeAgo(item.created_at)}</span>
                </div>
              </div>
              <ChevronRight className="w-3.5 h-3.5 shrink-0 text-slate-700 group-hover:text-slate-500 mt-0.5" />
            </div>
          </button>
        ))}
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center">
            <User className="w-3.5 h-3.5 text-slate-400" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-300">deniz-test-user</p>
            <p className="text-[10px] text-slate-600">Ücretsiz Plan</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

// Custom renderers that apply Tailwind classes to each Markdown element.
// Only used for assistant bubbles; user bubbles stay as plain text.
const mdComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-slate-100 mt-3 mb-1 first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-bold text-slate-100 mt-2.5 mb-1 first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-0.5 first:mt-0">{children}</h3>
  ),
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 last:mb-0 space-y-1 pl-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 last:mb-0 space-y-1 pl-4 list-decimal">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="flex items-start gap-2 text-slate-200">
      <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
      <span>{children}</span>
    </li>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-100">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-slate-300">{children}</em>
  ),
  code: ({ children }) => (
    <code className="bg-slate-700 text-indigo-300 rounded px-1 py-0.5 text-xs font-mono">
      {children}
    </code>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-indigo-500 pl-3 text-slate-400 italic my-2">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-slate-700 my-3" />,
};

function ChatBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5
          ${isUser ? "bg-indigo-600" : "bg-slate-700"}`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <BrainCircuit className="w-4 h-4 text-indigo-400" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[72%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm
          ${
            isUser
              ? "bg-indigo-600 text-white rounded-tr-sm"
              : "bg-slate-800 text-slate-200 rounded-tl-sm border border-slate-700"
          }`}
      >
        {isUser ? (
          msg.content
        ) : (
          <ReactMarkdown components={mdComponents}>{msg.content}</ReactMarkdown>
        )}
      </div>
    </div>
  );
}

const PROBLEM_TYPE_COLORS: Record<string, string> = {
  Growth: "text-emerald-400 bg-emerald-950 border-emerald-800",
  Cost: "text-amber-400 bg-amber-950 border-amber-800",
  Operational: "text-blue-400 bg-blue-950 border-blue-800",
  Technology: "text-purple-400 bg-purple-950 border-purple-800",
  Regulation: "text-orange-400 bg-orange-950 border-orange-800",
  Organizational: "text-pink-400 bg-pink-950 border-pink-800",
  Hybrid: "text-indigo-400 bg-indigo-950 border-indigo-800",
};

function ProblemTreePanel({ tree }: { tree: ProblemTree | null }) {
  if (!tree) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-8 gap-5">
        <div className="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
          <GitBranch className="w-8 h-8 text-slate-600" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-400 mb-1">Problem Ağacı</p>
          <p className="text-xs text-slate-600 leading-relaxed">
            Analiz tamamlandığında kök neden ağacınız burada görselleşecektir.
          </p>
        </div>
        <div className="w-full space-y-2">
          {[80, 60, 70].map((w, i) => (
            <div key={i} className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-slate-700 rounded-full" style={{ width: `${w}%` }} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const typeColor = tree.problem_type
    ? (PROBLEM_TYPE_COLORS[tree.problem_type] ?? PROBLEM_TYPE_COLORS.Hybrid)
    : PROBLEM_TYPE_COLORS.Hybrid;

  // Normalise root_causes — backend may return RootCause[] or legacy string[]
  const causes: RootCause[] = (tree.root_causes as Array<RootCause | string>).map((c) =>
    typeof c === "string" ? { main_cause: c, sub_causes: [] } : c
  );

  return (
    <div className="flex flex-col gap-4 p-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Target className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-bold text-slate-100">Problem Ağacı</h3>
      </div>

      {/* Problem Type badge */}
      {tree.problem_type && (
        <div className={`inline-flex items-center self-start rounded-full border px-3 py-1 text-[11px] font-semibold ${typeColor}`}>
          {tree.problem_type} Problemi
        </div>
      )}

      {/* Main problem */}
      <div className="rounded-xl bg-indigo-950 border border-indigo-800 p-4">
        <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider mb-1.5">
          Ana Problem
        </p>
        <p className="text-sm font-semibold text-slate-100 leading-snug">{tree.main_problem}</p>
      </div>

      {/* Industry + Confidence */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="rounded-xl bg-slate-800 border border-slate-700 p-3">
          <p className="text-[10px] text-slate-500 mb-0.5">Sektör</p>
          <p className="text-xs font-semibold text-slate-200">{tree.industry}</p>
        </div>
        <div className="rounded-xl bg-slate-800 border border-slate-700 p-3">
          <p className="text-[10px] text-slate-500 mb-0.5">Güven Skoru</p>
          <p className={`text-xs font-bold ${scoreColor(tree.confidence_score)}`}>
            {(tree.confidence_score * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Hierarchical root causes */}
      <div>
        <div className="flex items-center gap-1.5 mb-3">
          <TrendingDown className="w-4 h-4 text-rose-400" />
          <p className="text-xs font-semibold text-slate-300">Yapılandırılmış Problem Ağacı</p>
        </div>
        <div className="space-y-3">
          {causes.map((rc, i) => (
            <div key={i} className="rounded-lg border border-slate-700 overflow-hidden">
              {/* Main cause */}
              <div className="flex items-start gap-2.5 bg-slate-800 px-3 py-2.5">
                <span className="shrink-0 mt-0.5 w-4 h-4 rounded-full bg-rose-900 text-rose-400 text-[10px] font-bold flex items-center justify-center">
                  {i + 1}
                </span>
                <p className="text-xs font-semibold text-slate-200 leading-snug">{rc.main_cause}</p>
              </div>
              {/* Sub causes */}
              {rc.sub_causes.length > 0 && (
                <div className="bg-slate-900 px-3 py-2 space-y-1.5 border-t border-slate-700">
                  {rc.sub_causes.map((sc, j) => (
                    <div key={j} className="flex items-start gap-2">
                      <span className="shrink-0 mt-1 w-1.5 h-1.5 rounded-full bg-slate-600" />
                      <p className="text-[11px] text-slate-400 leading-snug">{sc}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Merhaba! Ben ENTRAPEER'in iş analizi ajanıyım. Şirketinizdeki karmaşık problemleri, krizleri veya stratejik soruları benimle paylaşabilirsiniz. Size kök neden analizi ve yapılandırılmış bir problem ağacı sunacağım.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [problemTree, setProblemTree] = useState<ProblemTree | null>(null);
  const [history, setHistory] = useState<Analysis[]>(MOCK_HISTORY);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [awaitingResponse, setAwaitingResponse] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Requests go to Next.js's own origin (/api/v1/…) which proxies to the
  // backend via the rewrites rule in next.config.ts. No CORS issue.
  const apiBase = "";

  async function sendMessage() {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      let data: {
        thread_id: string;
        messages: Message[];
        structured_problem: ProblemTree;
        current_step: string;
      };

      if (!awaitingResponse) {
        // First message → /analyze
        const res = await fetch(`${apiBase}/api/v1/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            history: messages,
            user_id: "deniz-test-user",
            thread_id: threadId ?? undefined,
          }),
        });
        data = await res.json();
        setThreadId(data.thread_id);
        setActiveThread(data.thread_id);
        // "awaiting_response" means discovery asked questions, user must answer
        if (data.current_step === "awaiting_response" || data.current_step === "discovery") {
          setAwaitingResponse(true);
        } else {
          setAwaitingResponse(false);
        }
      } else {
        // Follow-up → /respond
        const res = await fetch(`${apiBase}/api/v1/respond`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: threadId,
            message: text,
            user_id: "deniz-test-user",
          }),
        });
        data = await res.json();
        setAwaitingResponse(false);
      }

      // Show only new assistant messages
      const newAssistantMsgs = data.messages.filter((m) => m.role === "assistant");
      const lastAssistant = newAssistantMsgs[newAssistantMsgs.length - 1];
      if (lastAssistant) {
        setMessages((prev) => [...prev, lastAssistant]);
      }

      // Update problem tree if available
      if (data.structured_problem?.main_problem) {
        setProblemTree(data.structured_problem);
        // Push to history
        setHistory((prev) => {
          const exists = prev.find((h) => h.thread_id === data.thread_id);
          if (exists) return prev;
          return [
            {
              thread_id: data.thread_id,
              created_at: new Date().toISOString(),
              structured_problem: data.structured_problem,
            },
            ...prev,
          ];
        });
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Bağlantı hatası oluştu. Backend servisinin çalıştığından emin olunuz.",
        },
      ]);
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  }

  function handleNewChat() {
    setMessages([
      {
        role: "assistant",
        content:
          "Yeni bir analiz başlattınız. Şirketinizdeki problemi veya krizi benimle paylaşın.",
      },
    ]);
    setInput("");
    setThreadId(null);
    setActiveThread(null);
    setProblemTree(null);
    setAwaitingResponse(false);
  }

  function handleSelectThread(analysis: Analysis) {
    setActiveThread(analysis.thread_id);
    setThreadId(analysis.thread_id);
    setProblemTree(analysis.structured_problem);
    setMessages([
      {
        role: "assistant",
        content: `"${analysis.structured_problem.main_problem}" analizi yüklendi. Bu konuşmaya devam etmek için mesaj yazabilirsiniz.`,
      },
    ]);
    setAwaitingResponse(false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="h-screen flex overflow-hidden bg-slate-950">
      {/* ── Left: Sidebar ── */}
      <Sidebar
        history={history}
        activeThread={activeThread}
        onSelectThread={handleSelectThread}
        onNewChat={handleNewChat}
      />

      {/* ── Center: Chat ── */}
      <main className="flex-1 flex flex-col min-w-0 border-r border-slate-800">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-semibold text-slate-200">İş Analizi Asistanı</span>
          </div>
          {threadId && (
            <span className="text-[10px] font-mono text-slate-600 bg-slate-900 border border-slate-800 rounded px-2 py-1">
              {threadId.slice(0, 8)}…
            </span>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
          {messages.map((msg, i) => (
            <ChatBubble key={i} msg={msg} />
          ))}
          {isLoading && (
            <div className="flex gap-3 flex-row">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center shrink-0">
                <BrainCircuit className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="rounded-2xl rounded-tl-sm bg-slate-800 border border-slate-700 px-4 py-3">
                <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="shrink-0 px-6 py-4 border-t border-slate-800 bg-slate-950">
          {awaitingResponse && (
            <div className="mb-2 flex items-center gap-2 text-xs text-amber-400">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Keşif soruları yanıtınızı bekliyorum…</span>
            </div>
          )}
          <div className="flex items-end gap-3 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 shadow-lg shadow-black/30 focus-within:border-indigo-500 transition-colors">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                awaitingResponse
                  ? "Keşif sorularına cevaplarınızı yazın…"
                  : "Şirketinizdeki problemi veya soruyu yazın…"
              }
              className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 outline-none leading-relaxed
                         max-h-40 overflow-y-auto"
              style={{ fieldSizing: "content" } as React.CSSProperties}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center
                         bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600
                         text-white transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[10px] text-slate-700">
            Enter ile gönder · Shift+Enter ile yeni satır
          </p>
        </div>
      </main>

      {/* ── Right: Problem Tree ── */}
      <aside className="w-96 flex flex-col border-l border-slate-800 bg-slate-950 shrink-0">
        <header className="flex items-center gap-2 px-6 py-4 border-b border-slate-800 shrink-0">
          <Target className="w-4 h-4 text-rose-400" />
          <span className="text-sm font-semibold text-slate-200">Kök Neden Analizi</span>
        </header>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <ProblemTreePanel tree={problemTree} />
        </div>
      </aside>
    </div>
  );
}
