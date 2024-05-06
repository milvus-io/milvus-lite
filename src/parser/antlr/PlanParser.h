
// Generated from Plan.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  PlanParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, LT = 6, LE = 7, GT = 8, 
    GE = 9, EQ = 10, NE = 11, LIKE = 12, EXISTS = 13, ADD = 14, SUB = 15, 
    MUL = 16, DIV = 17, MOD = 18, POW = 19, SHL = 20, SHR = 21, BAND = 22, 
    BOR = 23, BXOR = 24, AND = 25, OR = 26, BNOT = 27, NOT = 28, IN = 29, 
    NIN = 30, EmptyTerm = 31, JSONContains = 32, JSONContainsAll = 33, JSONContainsAny = 34, 
    ArrayContains = 35, ArrayContainsAll = 36, ArrayContainsAny = 37, ArrayLength = 38, 
    BooleanConstant = 39, IntegerConstant = 40, FloatingConstant = 41, Identifier = 42, 
    StringLiteral = 43, JSONIdentifier = 44, Whitespace = 45, Newline = 46
  };

  enum {
    RuleExpr = 0
  };

  explicit PlanParser(antlr4::TokenStream *input);

  PlanParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~PlanParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class ExprContext; 

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ExprContext() = default;
    void copyFrom(ExprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  JSONIdentifierContext : public ExprContext {
  public:
    JSONIdentifierContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *JSONIdentifier();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ParensContext : public ExprContext {
  public:
    ParensContext(ExprContext *ctx);

    ExprContext *expr();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  StringContext : public ExprContext {
  public:
    StringContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *StringLiteral();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FloatingContext : public ExprContext {
  public:
    FloatingContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *FloatingConstant();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  JSONContainsAllContext : public ExprContext {
  public:
    JSONContainsAllContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *JSONContainsAll();
    antlr4::tree::TerminalNode *ArrayContainsAll();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LogicalOrContext : public ExprContext {
  public:
    LogicalOrContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *OR();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MulDivModContext : public ExprContext {
  public:
    MulDivModContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    antlr4::tree::TerminalNode *MOD();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IdentifierContext : public ExprContext {
  public:
    IdentifierContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *Identifier();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LikeContext : public ExprContext {
  public:
    LikeContext(ExprContext *ctx);

    ExprContext *expr();
    antlr4::tree::TerminalNode *LIKE();
    antlr4::tree::TerminalNode *StringLiteral();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LogicalAndContext : public ExprContext {
  public:
    LogicalAndContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *AND();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  EqualityContext : public ExprContext {
  public:
    EqualityContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NE();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BooleanContext : public ExprContext {
  public:
    BooleanContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *BooleanConstant();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ShiftContext : public ExprContext {
  public:
    ShiftContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *SHL();
    antlr4::tree::TerminalNode *SHR();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ReverseRangeContext : public ExprContext {
  public:
    ReverseRangeContext(ExprContext *ctx);

    antlr4::Token *op1 = nullptr;
    antlr4::Token *op2 = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *JSONIdentifier();
    std::vector<antlr4::tree::TerminalNode *> GT();
    antlr4::tree::TerminalNode* GT(size_t i);
    std::vector<antlr4::tree::TerminalNode *> GE();
    antlr4::tree::TerminalNode* GE(size_t i);

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BitOrContext : public ExprContext {
  public:
    BitOrContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *BOR();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AddSubContext : public ExprContext {
  public:
    AddSubContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *ADD();
    antlr4::tree::TerminalNode *SUB();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  RelationalContext : public ExprContext {
  public:
    RelationalContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *LE();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *GE();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArrayLengthContext : public ExprContext {
  public:
    ArrayLengthContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *ArrayLength();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *JSONIdentifier();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TermContext : public ExprContext {
  public:
    TermContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *NIN();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  JSONContainsContext : public ExprContext {
  public:
    JSONContainsContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *JSONContains();
    antlr4::tree::TerminalNode *ArrayContains();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  RangeContext : public ExprContext {
  public:
    RangeContext(ExprContext *ctx);

    antlr4::Token *op1 = nullptr;
    antlr4::Token *op2 = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *JSONIdentifier();
    std::vector<antlr4::tree::TerminalNode *> LT();
    antlr4::tree::TerminalNode* LT(size_t i);
    std::vector<antlr4::tree::TerminalNode *> LE();
    antlr4::tree::TerminalNode* LE(size_t i);

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryContext : public ExprContext {
  public:
    UnaryContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    ExprContext *expr();
    antlr4::tree::TerminalNode *ADD();
    antlr4::tree::TerminalNode *SUB();
    antlr4::tree::TerminalNode *BNOT();
    antlr4::tree::TerminalNode *NOT();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IntegerContext : public ExprContext {
  public:
    IntegerContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *IntegerConstant();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArrayContext : public ExprContext {
  public:
    ArrayContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  JSONContainsAnyContext : public ExprContext {
  public:
    JSONContainsAnyContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *JSONContainsAny();
    antlr4::tree::TerminalNode *ArrayContainsAny();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BitXorContext : public ExprContext {
  public:
    BitXorContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *BXOR();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ExistsContext : public ExprContext {
  public:
    ExistsContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *EXISTS();
    ExprContext *expr();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BitAndContext : public ExprContext {
  public:
    BitAndContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *BAND();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  EmptyTermContext : public ExprContext {
  public:
    EmptyTermContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    ExprContext *expr();
    antlr4::tree::TerminalNode *EmptyTerm();
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *NIN();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PowerContext : public ExprContext {
  public:
    PowerContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *POW();

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);

  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

