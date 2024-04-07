import ssl
from typing import List
import logging
from os import linesep
import smtplib
from imap_tools import MailBox, BaseMailBox, OR, AND

from agentopy import Action, ActionResult, IEnvironmentComponent, WithActionSpaceMixin, EntityInfo, State, IState

logger = logging.getLogger('[Component][Email]')


class Email(WithActionSpaceMixin, IEnvironmentComponent):
    """Implements an email environment component"""

    def __init__(self,
                 imap_address: str,
                 smtp_address: str,
                 imap_port: int,
                 smtp_port: int,
                 outbound_emails_whitelist: List[str] | None = None) -> None:
        super().__init__()
        self._imap_address: str = imap_address
        self._imap_port: int = imap_port
        self._smtp_address: str = smtp_address
        self._smtp_port: int = smtp_port
        self._outbound_emails_whitelist: List[str] | None = outbound_emails_whitelist
        self._mailbox: BaseMailBox | None = None  # type: ignore
        self._num_unseen_emails: int = -1
        self._smtp_server: smtplib.SMTP | None = None  # type: ignore

        self.action_space.register_actions(
            [
                Action(
                    "send_email", "Use this action to send an email to the specified address without attachements, one email at a time.", self.send_email, self.info()),
                Action(
                    "check_emails", "Use this action to check user's email inbox for emails received from 'email_from' containing text 'look_for_text'", self.check_emails, self.info())
            ]
        )

    async def _connect(self, login: str, password: str):
        self._mailbox: BaseMailBox = MailBox(self._imap_address, self._imap_port).login(
            login, password, initial_folder='INBOX')
        self._smtp_server: smtplib.SMTP = smtplib.SMTP(
            self._smtp_address, self._smtp_port)

        self._smtp_server.starttls(context=ssl.create_default_context())
        self._smtp_server.login(login, password)

    async def _disconnect(self):
        self._mailbox.logout()
        self._smtp_server.quit()

    async def send_email(self, *, to: str | List[str], subject: str, body: str, caller_context: IState, **kwargs) -> ActionResult:
        """
        Sends an email using smtp server
        """
        try:
            assert caller_context is not None, "Caller context is required"
            
            login = caller_context.get_item("login")
            password = caller_context.get_item("password")
            from_address = caller_context.get_item("from_address")
            
            await self._connect(login, password)

            message = f"From: {from_address}{linesep}To: {to}{linesep}Subject: {subject}{linesep}{linesep}{body}".encode()

            if isinstance(to, str):
                to_list = list(map(lambda s: s.strip(), to.split(',')))
            else:
                to_list = to

            if self._outbound_emails_whitelist is not None:
                to_list = list(
                    set(self._outbound_emails_whitelist).intersection(set(to_list)))

            if len(to_list) > 0:
                self._smtp_server.sendmail(
                    from_address, to_list, message)

            await self._disconnect()

            return ActionResult(value="Success", success=True)
        except Exception as error:
            logger.warn(f"Error sending email: {error}")
            await self._disconnect()
            return ActionResult(value=str(error), success=False)

    async def check_emails(self, *, look_for_text: str, from_email: str, caller_context: IState, **kwargs) -> ActionResult:
        """check user's email inbox for emails received from 'from_email' containing text 'look_for_text'"""

        try:
            assert caller_context is not None, "Caller context is required"
            
            login = caller_context.get_item("login")
            password = caller_context.get_item("password")
            
            await self._connect(login, password)

            condition = OR(subject=look_for_text, text=look_for_text)

            if from_email:
                condition = AND(
                    OR(subject=look_for_text, text=look_for_text), from_=from_email)

            emails = []

            for msg in self._mailbox.fetch(condition, reverse=True, limit=10, headers_only=True, mark_seen=True):
                emails.append(
                    f"From: {msg.from_}{linesep}Subject: {msg.subject}{linesep}Body: {msg.text}{linesep}"
                )

            emails_txt = linesep.join(emails)

            await self._disconnect()
            return ActionResult(value=f"No incoming emails from {from_email} with text {look_for_text}" if not emails else emails_txt, success=True)

        except Exception as error:
            logger.warn(f"Error checking emails: {error}")
            await self._disconnect()
            return ActionResult(value=str(error), success=False)

    async def observe(self, caller_context: IState) -> State:
        # await self._connect()
        # unseen_emails = len(
        #     list(self._mailbox.fetch(AND(seen=False), reverse=True, limit=10, headers_only=True, mark_seen=False)))
        state = State()
        new_emails = 0  # unseen_emails - self._num_unseen_emails
        state.set_item("status", {"num_new": new_emails})
        # await self._disconnect()
        return state
    
    async def tick(self) -> None:
        ...

    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__, 
            version="0.1.0", 
            params={}
        )
