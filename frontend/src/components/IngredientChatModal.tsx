import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Sparkles, AlertCircle, Bot, User, RefreshCw } from 'lucide-react';
import { AIIngredientCard, ChatResponse } from '../types';
import { sendChatQuestion } from '../api';

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  isGeneral?: boolean;
}

interface IngredientChatModalProps {
  isOpen: boolean;
  onClose: () => void;
  card: AIIngredientCard;
}

export const IngredientChatModal: React.FC<IngredientChatModalProps> = ({
  isOpen,
  onClose,
  card,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize suggestions and welcome message when card changes or modal opens
  useEffect(() => {
    if (isOpen) {
      const initialSuggestions = [
        `Where does ${card.name} come from?`,
        `How is ${card.name} made?`,
        `Why is ${card.name} used here?`,
      ];
      setSuggestedQuestions(initialSuggestions);
      setMessages([
        {
          id: 'welcome',
          sender: 'assistant',
          text: `Ask me anything about **${card.name}**. My answers are grounded strictly in this ingredient's verified record.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    }
  }, [isOpen, card.id, card.name]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async (questionText: string) => {
    const q = questionText.trim();
    if (!q || isLoading) return;

    setError(null);
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuestion('');
    setIsLoading(true);

    try {
      const res: ChatResponse = await sendChatQuestion(card.id || card.name, q, card);
      const isGeneral =
        res.source === 'general' ||
        res.answer.toLowerCase().includes('general information, not from the card');
      const botMsg: ChatMessage = {
        id: `b-${Date.now()}`,
        sender: 'assistant',
        text: res.answer,
        isGeneral,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, botMsg]);

      if (res.suggested_questions && res.suggested_questions.length > 0) {
        setSuggestedQuestions(res.suggested_questions);
      }
    } catch (err: any) {
      console.error('Chat error:', err);
      setError('Could not reach the explainer service right now.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4 bg-black/75 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, y: '100%' }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 220 }}
            className="w-full max-w-lg bg-zinc-900 border border-zinc-800 rounded-t-3xl sm:rounded-3xl shadow-2xl flex flex-col max-h-[85vh] h-[85vh] overflow-hidden"
          >
            {/* Header */}
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/40">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight">
                    Ask about {card.name}
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    Grounded strictly in this ingredient card
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                aria-label="Close Chat"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Messages Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3.5 text-xs">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-2.5 ${
                    msg.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {msg.sender === 'assistant' && (
                    <div className="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-300 flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="w-3.5 h-3.5" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-2xl px-3.5 py-2.5 ${
                      msg.sender === 'user'
                        ? 'bg-emerald-500 text-zinc-950 font-medium rounded-br-sm'
                        : 'bg-zinc-800 text-zinc-200 border border-zinc-700/60 rounded-bl-sm leading-relaxed'
                    }`}
                  >
                    {msg.isGeneral && (
                      <span className="inline-block px-2 py-0.5 rounded-full text-[9px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30 mb-1.5">
                        General information, not from the card
                      </span>
                    )}
                    <p className="whitespace-pre-wrap">{msg.text}</p>
                    <span
                      className={`block text-[9px] mt-1 ${
                        msg.sender === 'user' ? 'text-zinc-800/80' : 'text-zinc-500'
                      }`}
                    >
                      {msg.timestamp}
                    </span>
                  </div>
                  {msg.sender === 'user' && (
                    <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-300 flex items-center justify-center shrink-0 mt-0.5">
                      <User className="w-3.5 h-3.5" />
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-2.5 items-center text-zinc-400 py-1">
                  <div className="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-300 flex items-center justify-center shrink-0">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  </div>
                  <span className="text-xs italic">Reading ingredient card facts...</span>
                </div>
              )}

              {error && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Suggested Question Chips (3 follow-up prompts) */}
            {suggestedQuestions.length > 0 && (
              <div className="p-3 border-t border-zinc-800/70 bg-zinc-950/20 space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider block">
                  Suggested Questions
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {suggestedQuestions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSend(q)}
                      disabled={isLoading}
                      className="px-2.5 py-1 rounded-full bg-zinc-800 hover:bg-zinc-700 border border-zinc-700/60 text-zinc-300 hover:text-white text-[11px] transition-colors text-left disabled:opacity-50"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Bar */}
            <div className="p-3 border-t border-zinc-800 bg-zinc-950/80">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSend(inputQuestion);
                }}
                className="space-y-1.5"
              >
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    maxLength={300}
                    value={inputQuestion}
                    onChange={(e) => setInputQuestion(e.target.value)}
                    placeholder={`Ask a question about ${card.name}...`}
                    className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-3.5 py-2.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-indigo-500"
                    disabled={isLoading}
                  />
                  <button
                    type="submit"
                    disabled={!inputQuestion.trim() || isLoading}
                    className="p-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-zinc-950 font-bold transition-all shrink-0"
                    aria-label="Send Question"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex justify-between items-center px-1 text-[10px] text-zinc-500">
                  <span>Questions are answered from the card or general knowledge.</span>
                  <span>{inputQuestion.length}/300</span>
                </div>
              </form>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
