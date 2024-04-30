
// Generated from Plan.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  PlanLexer : public antlr4::Lexer {
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

  explicit PlanLexer(antlr4::CharStream *input);

  ~PlanLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

